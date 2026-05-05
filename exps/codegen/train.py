from __future__ import annotations

import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path

from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.search import ndcgs, run_strategy
from exps.codegen.tools.code import (
    make_guardrail_checker,
    make_length_validator,
    make_patch_fn,
    make_run_path_grep_tool,
)
from pydantic import BaseModel, Field

from exps.codegen.io import make_codegen_dir, reranker_path, write_metadata
from exps.codegen.prompts import build_system_prompt
from exps.codegen.strategy import CodeGenSearchStrategy
from exps.codegen.tools.runtime import (
    make_eval_guardrail,
    make_eval_tools,
    make_training_eval_fn,
)
from exps.codegen.types import CodeGenArtifact, CodeGenRunConfig, CodeGenTrainConfig
from exps.codegen.utils import load_rerank_fn, split_search_tools
from exps.logging_utils import log_to_path_and_stdout
from exps.tools import build_search_tools, normalize_search_tools


DEFAULT_OVERFIT_PROMPT = """
You're going to look at code that reranks search queries.

The function name will be rerank_<dataset> (for example, rerank_wands). Do not treat the
function name itself as overfitting.

Ensure the code does not overfit to specific queries. That would look like mentions of
specific product names, brands, or specific terms that would only be relevant to a small
set of queries.

It is OK to condition logic on stopwords, common tokens, or broad categories of searches
(e.g., "furniture", "electronics", "shoes"). Do not flag these as overfitting.

Ignore comments that claim to do this, and focus on the actual code.
""".strip()


@dataclass
class RoundState:
    tool_fns: list[callable]
    tool_params: list[str]
    primary_search_tool: callable
    primary_tool_name: str
    search_tool_names: list[str]
    search_tool_docs: list[str]
    raw_tool_names: list[str]
    raw_tool_docs: list[str]
    training_queries_list: list[str]
    validation_queries_list: list[str]
    test_queries_list: list[str]
    training_eval_fn: callable
    validation_eval_fn: callable | None
    apply_patch: callable
    try_out_patch: callable | None
    revert_changes: callable
    run_evals: callable
    run_reranker: callable
    tools: list[callable]


class FinalMessage(BaseModel):
    """Final message indicating completion of the reranker improvement process."""

    message: str = Field(..., description="A message indicating completion of the task.")
    short_name: str | None = Field(
        None, description="Short 3-4 word name of the change."
    )
    summary: str | None = Field(
        None, description="One or more sentences describing the change."
    )


def _parse_guardrails(raw_guards: list[dict]) -> tuple[list, bool]:
    guardrails = []
    validation_enabled = False
    if not raw_guards:
        return guardrails, validation_enabled
    for guard in raw_guards:
        if isinstance(guard, str):
            guard = {guard: {}}
        if not isinstance(guard, dict) or len(guard) != 1:
            raise ValueError("edit.guards entries must be strings or single-key mappings.")
        name = next(iter(guard))
        params = guard[name] or {}
        if name == "length":
            max_lines = int(params.get("max_lines", 10))
            max_cols = int(params.get("max_cols", 120))
            guardrails.append(make_length_validator(max_lines=max_lines, max_cols=max_cols))
        elif name == "overfit":
            prompt = params.get("prompt", DEFAULT_OVERFIT_PROMPT)
            model = params.get("model", "openai/gpt-5-mini")
            reasoning = params.get("reasoning", "medium")
            guardrails.append(
                make_guardrail_checker(
                    prompt=prompt,
                    model=model,
                    reasoning=reasoning,
                )
            )
        elif name == "validation":
            validation_enabled = True
        else:
            raise ValueError(f"Unknown edit guard: {name}")
    return guardrails, validation_enabled


def _make_rerank_name_guard(rerank_name: str) -> callable:
    def guard(code: str) -> str | None:
        try:
            local_vars: dict = {}
            exec(code, {}, local_vars)
        except Exception as exc:
            return f"Code failed to execute: {exc}"
        rerank_fn = local_vars.get(rerank_name)
        if not callable(rerank_fn):
            return f"Code must define a callable {rerank_name} function."
        return None

    guard.__doc__ = (
        f"Code must define a callable {rerank_name} function with the required signature."
    )
    return guard


def _validate_start_code(code: str, rerank_name: str, tool_fns: list[callable]) -> None:
    try:
        rerank_fn = load_rerank_fn(code, rerank_name)
    except Exception as exc:
        raise ValueError(f"start_code must define a callable {rerank_name} function: {exc}") from exc
    try:
        rerank_fn("test query", *tool_fns)
    except Exception as exc:
        raise ValueError(
            "start_code does not match configured tools; verify the rerank signature and search_tools."
        ) from exc


def _split_queries(
    *,
    base_queries: list[str],
    train_size: int,
    val_size: int,
    training_seed: int,
    validation_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if not base_queries:
        return [], [], []
    validation_queries = random.Random(validation_seed).sample(
        base_queries, min(val_size, len(base_queries))
    )
    remaining_queries = [q for q in base_queries if q not in set(validation_queries)]
    training_queries = random.Random(training_seed).sample(
        remaining_queries, min(train_size, len(remaining_queries))
    )
    test_queries = [q for q in base_queries if q not in set(training_queries)]
    return training_queries, validation_queries, test_queries


def _build_tool_state(
    *,
    corpus,
    dataset: str,
    device: str | None,
    normal_tool_config: list,
    raw_tool_config: list,
    rerank_name: str,
    code_path: Path,
    start_code_from_config: bool,
) -> tuple[
    list[callable],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    callable,
    str,
]:
    search_tools = build_search_tools(
        corpus,
        normal_tool_config,
        embeddings_device=device,
        dataset_name=dataset,
    )
    raw_tools = build_search_tools(
        corpus,
        raw_tool_config,
        embeddings_device=device,
        dataset_name=dataset,
        context="raw",
    )
    if not search_tools and not raw_tools:
        raise ValueError("Codegen requires at least one search tool.")
    tool_fns = search_tools + raw_tools
    if start_code_from_config:
        _validate_start_code(code_path.read_text(encoding="utf-8"), rerank_name, tool_fns)
    search_tool_names = [tool.__name__ for tool in search_tools]
    search_tool_docs = [tool.__doc__ or "" for tool in search_tools]
    raw_tool_names = [tool.__name__ for tool in raw_tools]
    raw_tool_docs = [tool.__doc__ or "" for tool in raw_tools]
    tool_params = search_tool_names + raw_tool_names
    if search_tools:
        primary_search_tool = search_tools[0]
        primary_tool_name = search_tool_names[0]
    else:
        primary_search_tool = raw_tools[0]
        primary_tool_name = raw_tool_names[0]
    return (
        tool_fns,
        tool_params,
        search_tool_names,
        search_tool_docs,
        raw_tool_names,
        raw_tool_docs,
        primary_search_tool,
        primary_tool_name,
    )


def _build_round_state(
    *,
    round_idx: int,
    corpus,
    judgments,
    dataset: str,
    device: str | None,
    workers: int,
    eval_cfg,
    train_config: CodeGenTrainConfig,
    run_config: CodeGenRunConfig,
    base_queries: list[str],
    train_size: int,
    val_size: int,
    guardrails: list[callable],
    validation_enabled: bool,
    rerank_name: str,
    code_path: Path,
    normal_tool_config: list,
    raw_tool_config: list,
    start_code_from_config: bool,
) -> RoundState:
    (
        tool_fns,
        tool_params,
        search_tool_names,
        search_tool_docs,
        raw_tool_names,
        raw_tool_docs,
        primary_search_tool,
        primary_tool_name,
    ) = _build_tool_state(
        corpus=corpus,
        dataset=dataset,
        device=device,
        normal_tool_config=normal_tool_config,
        raw_tool_config=raw_tool_config,
        rerank_name=rerank_name,
        code_path=code_path,
        start_code_from_config=start_code_from_config,
    )

    training_seed = eval_cfg.training_seed + round_idx
    validation_seed = eval_cfg.validation_seed + round_idx
    training_queries_list, validation_queries_list, test_queries_list = _split_queries(
        base_queries=base_queries,
        train_size=train_size,
        val_size=val_size,
        training_seed=training_seed,
        validation_seed=validation_seed,
    )
    training_eval_fn = make_training_eval_fn(
        corpus=corpus,
        judgments=judgments,
        tool_fns=tool_fns,
        rerank_name=rerank_name,
        seed=training_seed,
        num_queries=len(training_queries_list),
        queries=training_queries_list,
        workers=workers,
    )
    validation_eval_fn = None
    if validation_enabled:
        validation_eval_fn = make_eval_guardrail(
            corpus=corpus,
            judgments=judgments,
            tool_fns=tool_fns,
            rerank_name=rerank_name,
            seed=validation_seed,
            num_queries=len(validation_queries_list),
            queries=validation_queries_list,
            workers=workers,
        )
    apply_patch, try_out_patch, revert_changes = make_patch_fn(
        search_fn=primary_search_tool,
        corpus=corpus,
        code_dir=str(code_path.parent),
        tool_fns=tool_fns,
        module_name=code_path.stem,
        function_name=rerank_name,
        guardrail_fns=guardrails,
        training_eval_fn=training_eval_fn,
        validation_eval_fn=validation_eval_fn,
        eval_margin=eval_cfg.eval_margin,
    )
    grep_run_path = make_run_path_grep_tool(code_path.parent)

    run_evals, run_reranker = make_eval_tools(
        corpus=corpus,
        judgments=judgments,
        tool_fns=tool_fns,
        rerank_name=rerank_name,
        code_path=code_path,
        seed=training_seed,
        num_queries=len(training_queries_list),
        queries=training_queries_list,
        workers=workers,
    )
    tools = [
        *tool_fns[: len(search_tool_names)],
        apply_patch,
        run_reranker,
        run_evals,
        grep_run_path,
        revert_changes,
    ]
    if train_config.try_out_patch:
        tools.append(try_out_patch)

    if not code_path.exists():
        code_path.write_text(
            _start_code(
                rerank_name,
                run_config.top_k,
                tool_params=tool_params,
                primary_tool_name=primary_tool_name,
            ),
            encoding="utf-8",
        )

    return RoundState(
        tool_fns=tool_fns,
        tool_params=tool_params,
        primary_search_tool=primary_search_tool,
        primary_tool_name=primary_tool_name,
        search_tool_names=search_tool_names,
        search_tool_docs=search_tool_docs,
        raw_tool_names=raw_tool_names,
        raw_tool_docs=raw_tool_docs,
        training_queries_list=training_queries_list,
        validation_queries_list=validation_queries_list,
        test_queries_list=test_queries_list,
        training_eval_fn=training_eval_fn,
        validation_eval_fn=validation_eval_fn,
        apply_patch=apply_patch,
        try_out_patch=try_out_patch,
        revert_changes=revert_changes,
        run_evals=run_evals,
        run_reranker=run_reranker,
        tools=tools,
    )


def _start_code(
    rerank_name: str,
    top_k: int,
    *,
    tool_params: list[str],
    primary_tool_name: str,
) -> str:
    signature = ", ".join(["query", *tool_params, "**kwargs"])
    if "fielded_bm25" in primary_tool_name:
        call = (
            f"    docs = {primary_tool_name}"
            f"(query, fields=['title^9.3', 'description^4.1'], operator='or', top_k={top_k})\n"
        )
    elif primary_tool_name == "get_corpus":
        call = (
            "    corpus = get_corpus()\n"
            f"    docs = corpus.head({top_k}).to_dict('records')\n"
        )
    else:
        call = f"    docs = {primary_tool_name}(query, top_k={top_k})\n"
    return (
        f"def {rerank_name}({signature}):\n"
        f"{call}"
        "    return [str(doc['id']) for doc in docs]\n"
    )


def _resolve_path_continuation(path: Path) -> tuple[int, str | None]:
    previous_rounds = 0
    rounds_log = path / "rounds.jsonl"
    if rounds_log.exists():
        with rounds_log.open("r", encoding="utf-8") as handle:
            previous_rounds = sum(1 for _ in handle)

    round_files = list(path.glob("reranker_round_*.py"))
    if round_files:

        def _round_num(path: Path) -> int:
            name = path.stem
            try:
                return int(name.split("_round_")[-1])
            except ValueError:
                return -1

        last_round_path = max(round_files, key=_round_num)
        previous_rounds = max(_round_num(last_round_path), previous_rounds, 0)
        start_code = last_round_path.read_text(encoding="utf-8")
        return previous_rounds, start_code

    prior_reranker = path / "reranker.py"
    if prior_reranker.exists():
        start_code = prior_reranker.read_text(encoding="utf-8")
        return previous_rounds, start_code

    return previous_rounds, None


def train_codegen_strategy(
    *,
    strategy_name: str,
    dataset: str,
    corpus,
    judgments,
    params: dict,
    run_path: str | Path | None = None,
    device: str | None = None,
    workers: int = 1,
    report_num_queries: int | None = None,
    report_seed: int = 42,
    run_started_at: str | None = None,
) -> CodeGenArtifact:
    if judgments is None:
        raise ValueError("Codegen training requires judgments.")
    train_params = params.get("train") or {}
    run_params = params.get("run") or {}
    if "path" in run_params:
        raise ValueError("run.path is no longer supported; use strategy.path instead.")
    train_config = CodeGenTrainConfig.model_validate(train_params)
    run_config = CodeGenRunConfig.model_validate(run_params)

    output_dir: Path | None = None
    code_path: Path | None = None
    rerank_name = f"rerank_{dataset}"

    tool_config = train_config.search_tools or ["bm25"]
    normal_tool_config, raw_tool_config = split_search_tools(tool_config)
    normalized_tools = normalize_search_tools(normal_tool_config)

    if train_config.continue_from or "continue" in train_params:
        raise ValueError("train.continue is no longer supported; use strategy.path instead.")
    if train_params.get("start_with"):
        raise ValueError("start_with is no longer supported; use strategy.path instead.")
    start_code_from_config = False
    previous_rounds = 0
    start_code = None
    continue_from = None
    if run_path is not None:
        output_dir = Path(run_path).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        code_path = reranker_path(output_dir)
        previous_rounds, start_code = _resolve_path_continuation(output_dir)
        continue_from = str(output_dir) if previous_rounds > 0 else None
    else:
        if train_config.start_code:
            start_code = textwrap.dedent(train_config.start_code).lstrip()
            start_code_from_config = True
        if output_dir is None:
            output_dir = make_codegen_dir(dataset, strategy_name, run_started_at=run_started_at)
            code_path = reranker_path(output_dir)
    if start_code is not None:
        code_path.write_text(start_code, encoding="utf-8")

    log_path = output_dir / "codegen.log"
    train_logger = log_to_path_and_stdout("codegen.train", log_path)
    log_to_path_and_stdout("code", log_path)
    log_to_path_and_stdout("eval", log_path)

    eval_cfg = train_config.eval
    query_cols = ["query"]
    if "query_id" in judgments.columns:
        query_cols.append("query_id")
    available_queries = judgments[query_cols].drop_duplicates()["query"].tolist()
    if report_num_queries is not None:
        report_count = min(report_num_queries, len(available_queries))
        base_queries = random.Random(report_seed).sample(available_queries, report_count)
    else:
        base_queries = list(available_queries)
        report_count = len(base_queries)

    train_size = int(len(base_queries) * eval_cfg.train_query_fraction)
    val_size = int(len(base_queries) * eval_cfg.validation_query_fraction)
    if base_queries and eval_cfg.train_query_fraction > 0 and train_size == 0:
        train_size = 1
    if base_queries and eval_cfg.validation_query_fraction > 0 and val_size == 0:
        val_size = 1

    guardrails, validation_enabled = _parse_guardrails(train_config.edit.guards)
    guardrails.append(_make_rerank_name_guard(rerank_name))

    messages: list[str] = []
    round_summaries: list[dict] = []
    round_ndcgs: list[float] = []
    round_test_ndcgs: list[float] = []
    rounds_log_path = output_dir / "rounds.jsonl"
    refresh_every = train_config.refresh_every or train_config.rounds
    if refresh_every <= 0:
        raise ValueError("refresh_every must be >= 1")
    round_state: RoundState | None = None
    total_rounds = previous_rounds + train_config.rounds
    if previous_rounds == 0 and not rounds_log_path.exists():
        round_state = _build_round_state(
            round_idx=previous_rounds,
            corpus=corpus,
            judgments=judgments,
            dataset=dataset,
            device=device,
            workers=workers,
            eval_cfg=eval_cfg,
            train_config=train_config,
            run_config=run_config,
            base_queries=base_queries,
            train_size=train_size,
            val_size=val_size,
            guardrails=guardrails,
            validation_enabled=validation_enabled,
            rerank_name=rerank_name,
            code_path=code_path,
            normal_tool_config=normal_tool_config,
            raw_tool_config=raw_tool_config,
            start_code_from_config=start_code_from_config,
        )
        start_code_from_config = False
        baseline_code = code_path.read_text(encoding="utf-8")
        baseline_strategy = CodeGenSearchStrategy(
            corpus,
            search_fn=round_state.primary_search_tool,
            tool_fns=round_state.tool_fns,
            code=baseline_code,
            rerank_name=rerank_name,
            workers=workers,
        )
        baseline_results = run_strategy(
            baseline_strategy,
            judgments,
            queries=base_queries,
            seed=report_seed,
            cache=False,
        )
        baseline_ndcg = float(ndcgs(baseline_results).mean()) if not baseline_results.empty else 0.0
        baseline_test_ndcg = 0.0
        if round_state.test_queries_list:
            baseline_test_results = run_strategy(
                baseline_strategy,
                judgments,
                queries=round_state.test_queries_list,
                seed=report_seed,
                cache=False,
            )
            baseline_test_ndcg = (
                float(ndcgs(baseline_test_results).mean())
                if not baseline_test_results.empty
                else 0.0
            )
        round_ndcgs.append(baseline_ndcg)
        round_test_ndcgs.append(baseline_test_ndcg)
        baseline_record = {
            "round": 0,
            "short_name": "baseline",
            "summary": "Initial reranker baseline",
            "message": None,
            "mean_ndcg": baseline_ndcg,
            "mean_ndcg_test": baseline_test_ndcg,
            "training_query_count": len(round_state.training_queries_list),
            "validation_query_count": len(round_state.validation_queries_list),
            "test_query_count": len(round_state.test_queries_list),
        }
        round_summaries.append(baseline_record)
        rounds_log_path.write_text(json.dumps(baseline_record) + "\n", encoding="utf-8")
    for round_idx in range(previous_rounds, total_rounds):
        refresh_round = (round_idx - previous_rounds) % refresh_every == 0
        if refresh_round:
            round_state = _build_round_state(
                round_idx=round_idx,
                corpus=corpus,
                judgments=judgments,
                dataset=dataset,
                device=device,
                workers=workers,
                eval_cfg=eval_cfg,
                train_config=train_config,
                run_config=run_config,
                base_queries=base_queries,
                train_size=train_size,
                val_size=val_size,
                guardrails=guardrails,
                validation_enabled=validation_enabled,
                rerank_name=rerank_name,
                code_path=code_path,
                normal_tool_config=normal_tool_config,
                raw_tool_config=raw_tool_config,
                start_code_from_config=start_code_from_config,
            )
            start_code_from_config = False
        if round_state is None:
            raise ValueError("Codegen round state was not initialized.")

        train_logger.info("Starting training round %s/%s...", round_idx + 1, total_rounds)
        code = code_path.read_text(encoding="utf-8")
        system_prompt = build_system_prompt(
            train_config.system_prompt,
            dataset=dataset,
            rerank_name=rerank_name,
            search_tool_names=round_state.search_tool_names,
            search_tool_docs=round_state.search_tool_docs,
            raw_tool_names=round_state.raw_tool_names,
            raw_tool_docs=round_state.raw_tool_docs,
            rerank_params=["query", *round_state.tool_params, "**kwargs"],
            code=code,
        )
        agent = OpenAIAgent(
            tools=round_state.tools,
            model="openai/" + train_config.model,
            response_model=FinalMessage,
            reasoning_level=train_config.reasoning,
        )
        inputs = [{"role": "system", "content": system_prompt}]
        resp: FinalMessage | None = agent.loop(inputs=inputs)
        if resp is not None:
            messages.append(resp.message)
        code = code_path.read_text(encoding="utf-8")
        codegen_strategy = CodeGenSearchStrategy(
            corpus,
            search_fn=round_state.primary_search_tool,
            tool_fns=round_state.tool_fns,
            code=code,
            rerank_name=rerank_name,
            workers=workers,
        )
        results_codegen = run_strategy(
            codegen_strategy,
            judgments,
            queries=base_queries,
            seed=report_seed,
            cache=False,
        )
        mean_ndcg = float(ndcgs(results_codegen).mean()) if not results_codegen.empty else 0.0
        mean_test_ndcg = 0.0
        if round_state.test_queries_list:
            results_test = run_strategy(
                codegen_strategy,
                judgments,
                queries=round_state.test_queries_list,
                seed=report_seed,
                cache=False,
            )
            mean_test_ndcg = (
                float(ndcgs(results_test).mean()) if not results_test.empty else 0.0
            )
        round_ndcgs.append(mean_ndcg)
        round_test_ndcgs.append(mean_test_ndcg)
        train_logger.info(
            "Round %s/%s mean NDCG: %.4f (test %.4f)",
            round_idx + 1,
            total_rounds,
            mean_ndcg,
            mean_test_ndcg,
        )
        round_record = {
            "round": round_idx + 1,
            "short_name": resp.short_name if resp else None,
            "summary": resp.summary if resp else None,
            "message": resp.message if resp else None,
            "mean_ndcg": mean_ndcg,
            "mean_ndcg_test": mean_test_ndcg,
            "training_query_count": len(round_state.training_queries_list),
            "validation_query_count": len(round_state.validation_queries_list),
            "test_query_count": len(round_state.test_queries_list),
        }
        round_summaries.append(round_record)
        rounds_log_path.write_text(
            (rounds_log_path.read_text(encoding="utf-8") if rounds_log_path.exists() else "")
            + json.dumps(round_record) + "\n",
            encoding="utf-8",
        )
        round_code_path = output_dir / f"reranker_round_{round_idx + 1}.py"
        round_code_path.write_text(code, encoding="utf-8")

    final_code = code_path.read_text(encoding="utf-8")
    if round_state is None:
        raise ValueError("Codegen round state was not initialized.")
    metadata = {
        "dataset": dataset,
        "strategy_name": strategy_name,
        "rerank_name": rerank_name,
        "model": train_config.model,
        "rounds": total_rounds,
        "rounds_added": train_config.rounds,
        "continued_from": continue_from,
        "previous_rounds": previous_rounds,
        "refresh_every": refresh_every,
        "search_tools": normalized_tools,
        "messages": messages,
        "round_summaries": round_summaries,
        "round_ndcgs": round_ndcgs,
        "round_test_ndcgs": round_test_ndcgs,
        "training_seed": eval_cfg.training_seed,
        "validation_seed": eval_cfg.validation_seed,
        "train_query_fraction": eval_cfg.train_query_fraction,
        "validation_query_fraction": eval_cfg.validation_query_fraction,
        "num_training_queries": len(round_state.training_queries_list),
        "num_validation_queries": len(round_state.validation_queries_list),
        "base_query_count": len(base_queries),
        "report_seed": report_seed,
        "report_num_queries": report_num_queries,
        "eval_margin": eval_cfg.eval_margin,
    }
    write_metadata(output_dir, metadata)
    return CodeGenArtifact(
        path=output_dir,
        reranker_path=code_path,
        code=final_code,
        metadata=metadata,
        search_fn=round_state.primary_search_tool,
        tool_fns=round_state.tool_fns,
    )
