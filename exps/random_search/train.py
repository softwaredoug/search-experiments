from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cheat_at_search.search import ndcgs, run_strategy
from pydantic import BaseModel, ConfigDict, Field

from exps.strategies.bm25 import BM25Strategy


_PLACEHOLDER = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_BM25_PARAMS = {"k1", "b", "title_boost", "description_boost"}


class RandomSearchEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    seed: int = 1234


class RandomSearchTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rounds: int = 100
    search_seed: int = 42
    search_space: dict[str, dict[str, Any]] = Field(default_factory=dict)
    objective: str = "ndcg"
    eval: RandomSearchEvalConfig = Field(default_factory=RandomSearchEvalConfig)


@dataclass
class RandomSearchArtifact:
    path: Path
    metadata: dict[str, Any]


def _find_placeholders(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_find_placeholders(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_find_placeholders(item) for item in value))
    if isinstance(value, str):
        match = _PLACEHOLDER.fullmatch(value)
        if match:
            return {match.group(1)}
        if "{" in value or "}" in value:
            raise ValueError(f"Invalid parameter placeholder: {value!r}")
    return set()


def _as_finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Search value for {name!r} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Search value for {name!r} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"Search value for {name!r} must be finite.")
    return number


def _validate_bm25_params(params: dict[str, Any]) -> dict[str, float]:
    unknown = set(params) - _BM25_PARAMS
    if unknown:
        raise ValueError(
            "Unsupported BM25 parameter(s): " + ", ".join(sorted(unknown))
        )
    missing = {"k1", "b"} - set(params)
    if missing:
        raise ValueError("BM25 params require: " + ", ".join(sorted(missing)))

    numeric = {name: _as_finite_number(value, name=name) for name, value in params.items()}
    if numeric["k1"] <= 0:
        raise ValueError("BM25 k1 must be greater than zero.")
    if not 0 <= numeric["b"] <= 1:
        raise ValueError("BM25 b must be between zero and one.")
    for name in ("title_boost", "description_boost"):
        if name in numeric and numeric[name] < 0:
            raise ValueError(f"BM25 {name} must be non-negative.")
    return numeric


def _validate_search_space(
    placeholders: set[str], search_space: dict[str, dict[str, Any]]
) -> None:
    missing = placeholders - set(search_space)
    unused = set(search_space) - placeholders
    if missing or unused:
        parts = []
        if missing:
            parts.append("missing search-space entries for: " + ", ".join(sorted(missing)))
        if unused:
            parts.append("unused search-space entries: " + ", ".join(sorted(unused)))
        raise ValueError("; ".join(parts))

    for name, spec in search_space.items():
        if name not in _BM25_PARAMS:
            raise ValueError(f"Unsupported BM25 search parameter: {name!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"Search-space entry for {name!r} must be a mapping.")
        distribution = spec.get("distribution")
        if distribution == "uniform":
            if set(spec) != {"distribution", "min", "max"}:
                raise ValueError(
                    f"Uniform search-space entry for {name!r} requires only min and max."
                )
            low = _as_finite_number(spec["min"], name=name)
            high = _as_finite_number(spec["max"], name=name)
            if low > high:
                raise ValueError(f"Search-space min exceeds max for {name!r}.")
            _validate_bm25_params({**({"k1": 1.0, "b": 0.5}), name: low})
            _validate_bm25_params({**({"k1": 1.0, "b": 0.5}), name: high})
        elif distribution == "choice":
            if set(spec) != {"distribution", "values"}:
                raise ValueError(
                    f"Choice search-space entry for {name!r} requires values."
                )
            values = spec["values"]
            if not isinstance(values, list) or not values:
                raise ValueError(f"Choice search-space values for {name!r} must be a non-empty list.")
            for value in values:
                numeric = _as_finite_number(value, name=name)
                _validate_bm25_params({**({"k1": 1.0, "b": 0.5}), name: numeric})
        else:
            raise ValueError(
                f"Unsupported distribution for {name!r}: {distribution!r}. "
                "Supported distributions are 'uniform' and 'choice'."
            )


def materialize_engine_config(
    engine_config: dict[str, Any], search_space: dict[str, dict[str, Any]], rng: random.Random
) -> dict[str, Any]:
    if not isinstance(engine_config, dict) or engine_config.get("base") != "bm25":
        base = engine_config.get("base") if isinstance(engine_config, dict) else None
        raise ValueError(f"Only retrieval_engine.base 'bm25' is supported, got {base!r}.")
    if set(engine_config) - {"base", "params"}:
        raise ValueError("retrieval_engine supports only 'base' and 'params'.")
    engine_params = engine_config.get("params")
    if not isinstance(engine_params, dict):
        raise ValueError("retrieval_engine.params must be a mapping.")

    placeholders = _find_placeholders(engine_params)
    _validate_search_space(placeholders, search_space)
    sampled: dict[str, float] = {}
    for name, spec in search_space.items():
        if spec["distribution"] == "uniform":
            sampled[name] = rng.uniform(float(spec["min"]), float(spec["max"]))
        else:
            sampled[name] = _as_finite_number(rng.choice(spec["values"]), name=name)

    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            match = _PLACEHOLDER.fullmatch(value)
            return sampled[match.group(1)] if match else value
        return value

    resolved_params = _validate_bm25_params(replace(engine_params))
    return {"base": "bm25", "params": resolved_params}


def split_queries(
    queries: list[str], *, train_fraction: float, validation_fraction: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train.eval.train_fraction must be between zero and one.")
    if not 0 < validation_fraction < 1:
        raise ValueError("train.eval.validation_fraction must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one.")
    if len(queries) < 3:
        raise ValueError("Random search requires at least three judged queries to split.")

    fractions = (
        train_fraction,
        validation_fraction,
        1 - train_fraction - validation_fraction,
    )
    raw = [fraction * len(queries) for fraction in fractions]
    allocations = [math.floor(value) for value in raw]
    left = len(queries) - sum(allocations)
    order = sorted(
        range(3),
        key=lambda idx: (raw[idx] - allocations[idx], -idx),
        reverse=True,
    )
    for idx in order[:left]:
        allocations[idx] += 1
    for idx, size in enumerate(allocations):
        if size == 0:
            donors = [donor for donor, count in enumerate(allocations) if count > 1]
            if not donors:
                raise ValueError(
                    "Query split fractions cannot produce non-empty train, "
                    "validation and test sets."
                )
            donor = max(donors, key=lambda donor_idx: allocations[donor_idx])
            allocations[donor] -= 1
            allocations[idx] = 1
    sizes = allocations

    shuffled = list(queries)
    random.Random(seed).shuffle(shuffled)
    train_end = sizes[0]
    validation_end = train_end + sizes[1]
    return shuffled[:train_end], shuffled[train_end:validation_end], shuffled[validation_end:]


def _mean_ndcg(*, strategy, judgments, queries: list[str], seed: int) -> float:
    graded = run_strategy(
        strategy,
        judgments,
        queries=queries,
        seed=seed,
        cache=False,
        show_progress=False,
    )
    values = ndcgs(graded)
    return float(values.mean()) if not values.empty else 0.0


def _read_trials(path: Path) -> list[dict[str, Any]]:
    trials_path = path / "trials.jsonl"
    if not trials_path.exists():
        return []
    trials = []
    for line in trials_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            trials.append(json.loads(line))
    return trials


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_random_search_strategy(
    *,
    strategy_name: str,
    dataset: str,
    corpus,
    judgments,
    params: dict,
    run_path: str | Path,
    workers: int = 1,
    report_num_queries: int | None = None,
    report_seed: int = 42,
) -> RandomSearchArtifact:
    run_path = Path(run_path).expanduser()
    if not run_path.exists() or not run_path.is_dir():
        raise ValueError(f"Random-search run path must be an existing directory: {run_path}")
    if judgments is None or "query" not in judgments.columns:
        raise ValueError("Random search requires query judgments.")

    train_config = RandomSearchTrainConfig.model_validate(params.get("train") or {})
    if train_config.objective != "ndcg":
        raise ValueError("Only the 'ndcg' objective is supported.")
    if train_config.rounds < 1:
        raise ValueError("Random-search rounds must be at least one.")
    engine_template = params.get("retrieval_engine") or {}
    if not isinstance(engine_template, dict) or engine_template.get("base") != "bm25":
        raise ValueError("Only retrieval_engine.base 'bm25' is supported.")
    if set(engine_template) - {"base", "params"}:
        raise ValueError("retrieval_engine supports only 'base' and 'params'.")
    engine_params = engine_template.get("params")
    if not isinstance(engine_params, dict):
        raise ValueError("retrieval_engine.params must be a mapping.")
    placeholders = _find_placeholders(engine_params)
    if not placeholders:
        raise ValueError("Random search requires at least one parameter placeholder.")
    _validate_search_space(placeholders, train_config.search_space)
    static_params = {
        name: value for name, value in engine_params.items()
        if not (isinstance(value, str) and _PLACEHOLDER.fullmatch(value))
    }
    # Validate all fixed parameters before starting trials.
    fixed_with_defaults = {"k1": 1.2, "b": 0.75, **static_params}
    _validate_bm25_params(fixed_with_defaults)

    metadata_path = run_path / "metadata.json"
    previous_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    effective_num_queries = report_num_queries
    if previous_metadata:
        previous_num_queries = previous_metadata.get("report_num_queries")
        if report_num_queries is None:
            effective_num_queries = previous_num_queries
        elif report_num_queries != previous_num_queries:
            raise ValueError("Cannot continue random search with a different num_queries value.")
        if previous_metadata.get("query_sample_seed", report_seed) != report_seed:
            raise ValueError("Cannot continue random search with a different query-sampling seed.")

    query_values = judgments["query"].drop_duplicates().tolist()
    saved_queries = previous_metadata.get("base_queries") if previous_metadata else None
    if saved_queries:
        if not set(saved_queries).issubset(set(query_values)):
            raise ValueError("Saved random-search queries are no longer present in judgments.")
        query_values = saved_queries
    elif effective_num_queries is not None:
        if effective_num_queries < 1:
            raise ValueError("num_queries must be at least one.")
        query_values = random.Random(report_seed).sample(
            query_values, min(effective_num_queries, len(query_values))
        )
    if not query_values:
        raise ValueError("Random search requires at least one judged query.")

    train_queries, validation_queries, test_queries = split_queries(
        query_values,
        train_fraction=train_config.eval.train_fraction,
        validation_fraction=train_config.eval.validation_fraction,
        seed=train_config.eval.seed,
    )
    signature = {
        "dataset": dataset,
        "strategy_name": strategy_name,
        "engine_template": engine_template,
        "search_space": train_config.search_space,
        "search_seed": train_config.search_seed,
        "objective": train_config.objective,
        "eval": train_config.eval.model_dump(),
        "base_queries": query_values,
        "report_num_queries": effective_num_queries,
        "report_seed": report_seed,
    }
    signature_json = json.dumps(signature, sort_keys=True, default=str)

    if previous_metadata and previous_metadata.get("configuration_signature") != signature_json:
        raise ValueError("Cannot continue random search with a different configuration or query split.")

    trials = _read_trials(run_path)
    with (run_path / "trials.jsonl").open("a", encoding="utf-8") as trial_log:
        for _ in range(train_config.rounds):
            trial_number = len(trials) + 1
            rng = random.Random(train_config.search_seed + trial_number - 1)
            materialized_engine = None
            try:
                materialized_engine = materialize_engine_config(
                    engine_template, train_config.search_space, rng
                )
                candidate = BM25Strategy.build(
                    materialized_engine["params"],
                    corpus=corpus,
                    workers=workers,
                )
                train_ndcg = _mean_ndcg(
                    strategy=candidate,
                    judgments=judgments,
                    queries=train_queries,
                    seed=report_seed,
                )
                validation_ndcg = _mean_ndcg(
                    strategy=candidate,
                    judgments=judgments,
                    queries=validation_queries,
                    seed=report_seed,
                )
                record = {
                    "round": trial_number,
                    "params": materialized_engine["params"],
                    "train_ndcg": train_ndcg,
                    "validation_ndcg": validation_ndcg,
                    "status": "completed",
                }
            except Exception as exc:
                record = {
                    "round": trial_number,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if materialized_engine is not None:
                    record["params"] = materialized_engine["params"]
            trial_log.write(json.dumps(record, sort_keys=True) + "\n")
            trial_log.flush()
            trials.append(record)

    successful_trials = [trial for trial in trials if trial.get("status") == "completed"]
    if not successful_trials:
        raise ValueError("Random search produced no successful candidates.")
    winner = max(successful_trials, key=lambda trial: (trial["validation_ndcg"], -trial["round"]))
    winning_params = winner["params"]
    winning_strategy = BM25Strategy.build(
        winning_params, corpus=corpus, workers=workers
    )
    test_ndcg = _mean_ndcg(
        strategy=winning_strategy,
        judgments=judgments,
        queries=test_queries,
        seed=report_seed,
    )

    best_engine = {"base": "bm25", "params": winning_params}
    _write_json(run_path / "best_params.json", {"retrieval_engine": best_engine})
    metadata = {
        "strategy_type": "random_search",
        "strategy_name": strategy_name,
        "dataset": dataset,
        "objective": train_config.objective,
        "rounds": len(trials),
        "rounds_added": train_config.rounds,
        "search_seed": train_config.search_seed,
        "query_sample_seed": report_seed,
        "query_split_seed": train_config.eval.seed,
        "report_num_queries": effective_num_queries,
        "base_queries": query_values,
        "base_query_count": len(query_values),
        "train_query_count": len(train_queries),
        "validation_query_count": len(validation_queries),
        "test_query_count": len(test_queries),
        "winning_round": winner["round"],
        "winning_params": winning_params,
        "train_ndcg": winner["train_ndcg"],
        "validation_ndcg": winner["validation_ndcg"],
        "test_ndcg": test_ndcg,
        "configuration": {
            "retrieval_engine": engine_template,
            "search_space": train_config.search_space,
            "eval": train_config.eval.model_dump(),
        },
        "configuration_signature": signature_json,
    }
    _write_json(metadata_path, metadata)
    return RandomSearchArtifact(path=run_path, metadata=metadata)
