from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.search import ndcgs, run_strategy
from exps.codegen.tools.code import (
    make_guardrail_checker,
    make_length_validator,
    make_patch_fn,
)
from pydantic import BaseModel, Field

from exps.codegen.io import find_latest_codegen_run, make_codegen_dir, reranker_path, write_metadata
from exps.codegen.prompts import build_system_prompt
from exps.codegen.strategy import CodeGenSearchStrategy
from exps.codegen.tools.runtime import (
    make_eval_guardrail,
    make_eval_tools,
    make_training_eval_fn,
)
from exps.codegen.types import CodeGenArtifact, CodeGenRunConfig, CodeGenTrainConfig
from exps.tools import build_search_tools, normalize_search_tools


DEFAULT_OVERFIT_PROMPT = """
You're going to look at code that reranks search queries.

The function name will be rerank_<dataset> (for example, rerank_wands). Do not treat the
function name itself as overfitting.

Ensure the code does not overfit to specific queries. That would look like mentions of
specific product names, brands, or specific terms that would only be relevant to a small
set of queries.

Ignore comments that claim to do this, and focus on the actual code.
""".strip()


class FinalMessage(BaseModel):
    """Final message indicating completion of the reranker improvement process."""

    message: str = Field(..., description="A message indicating completion of the task.")
    short_name: str | None = Field(
        None, description="Short 3-4 word name of the change."
    )
    summary: str | None = Field(
        None, description="One or more sentences describing the change."
    )


def _parse_guardrails(raw_guards: list[dict]) -> list:
    guardrails = []
    if not raw_guards:
        raw_guards = [
            {"length": {"max_lines": 10, "max_cols": 120}},
            {"overfit": {"prompt": DEFAULT_OVERFIT_PROMPT}},
        ]
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
            guardrails.append(make_guardrail_checker(prompt=prompt, model=model))
        else:
            raise ValueError(f"Unknown edit guard: {name}")
    return guardrails


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


def _start_code(
    rerank_name: str,
    top_k: int,
    *,
    tool_params: list[str],
    primary_tool_name: str,
) -> str:
    signature = ", ".join([*tool_params, "query"])
    if "fielded_bm25" in primary_tool_name:
        call = (
            f"    docs = {primary_tool_name}"
            f"(query, field_to_search='title', operator='or', top_k={top_k})\n"
        )
    else:
        call = f"    docs = {primary_tool_name}(query, top_k={top_k})\n"
    return (
        f"def {rerank_name}({signature}):\n"
        f"{call}"
        "    return [str(doc['id']) for doc in docs]\n"
    )


def train_codegen_strategy(
    *,
    strategy_name: str,
    dataset: str,
    corpus,
    judgments,
    params: dict,
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
    train_config = CodeGenTrainConfig.model_validate(train_params)
    run_config = CodeGenRunConfig.model_validate(run_params)

    output_dir: Path | None = None
    code_path: Path | None = None
    rerank_name = f"rerank_{dataset}"

    tool_config = train_config.search_tools or ["bm25"]
    normalized_tools = normalize_search_tools(tool_config)
    search_tools = build_search_tools(
        corpus,
        tool_config,
        embeddings_device=device,
        dataset_name=dataset,
    )
    if not search_tools:
        raise ValueError("Codegen requires at least one search tool.")
    search_tool_names = [tool.__name__ for tool in search_tools]
    search_tool_docs = [tool.__doc__ or "" for tool in search_tools]
    tool_params = search_tool_names.copy()
    primary_search_tool = search_tools[0]
    primary_tool_name = search_tool_names[0]

    continue_from = train_config.continue_from
    continue_path: Path | None = None
    previous_rounds = 0
    start_code: str | None = None
    if continue_from:
        if isinstance(continue_from, str) and continue_from != "latest":
            continue_path = Path(continue_from).expanduser()
        else:
            continue_path = find_latest_codegen_run(dataset, strategy_name)
        if continue_path is None:
            raise FileNotFoundError(
                f"No prior codegen runs found for dataset={dataset} strategy={strategy_name}."
            )
        if not continue_path.exists():
            raise FileNotFoundError(f"continue path not found: {continue_path}")

        rounds_log = continue_path / "rounds.jsonl"
        if rounds_log.exists():
            with rounds_log.open("r", encoding="utf-8") as handle:
                previous_rounds = sum(1 for _ in handle)

        round_files = list(continue_path.glob("reranker_round_*.py"))
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
        else:
            prior_reranker = continue_path / "reranker.py"
            if not prior_reranker.exists():
                raise FileNotFoundError(f"No reranker code found in {continue_path}")
            start_code = prior_reranker.read_text(encoding="utf-8")
    elif train_config.start_with:
        start_path = Path(train_config.start_with).expanduser()
        if not start_path.exists():
            raise FileNotFoundError(f"start_with path not found: {start_path}")
        start_code = start_path.read_text(encoding="utf-8")
    if output_dir is None:
        output_dir = make_codegen_dir(dataset, strategy_name, run_started_at=run_started_at)
        code_path = reranker_path(output_dir)
    if continue_path is not None:
        rounds_log = continue_path / "rounds.jsonl"
        if rounds_log.exists():
            shutil.copyfile(rounds_log, output_dir / "rounds.jsonl")
    if start_code is not None:
        code_path.write_text(start_code, encoding="utf-8")
    elif not code_path.exists():
        code_path.write_text(
            _start_code(
                rerank_name,
                run_config.top_k,
                tool_params=tool_params,
                primary_tool_name=primary_tool_name,
            ),
            encoding="utf-8",
        )

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

    validation_queries_list = random.Random(eval_cfg.validation_seed).sample(
        base_queries, min(val_size, len(base_queries))
    )
    remaining_queries = [q for q in base_queries if q not in set(validation_queries_list)]
    training_queries_list = random.Random(eval_cfg.training_seed).sample(
        remaining_queries, min(train_size, len(remaining_queries))
    )
    training_eval_fn = make_training_eval_fn(
        corpus=corpus,
        judgments=judgments,
        tool_fns=search_tools,
        rerank_name=rerank_name,
        seed=eval_cfg.training_seed,
        num_queries=len(training_queries_list),
        queries=training_queries_list,
        workers=workers,
    )
    validation_eval_fn = make_eval_guardrail(
        corpus=corpus,
        judgments=judgments,
        tool_fns=search_tools,
        rerank_name=rerank_name,
        seed=eval_cfg.validation_seed,
        num_queries=len(validation_queries_list),
        queries=validation_queries_list,
        workers=workers,
    )

    guardrails = _parse_guardrails(train_config.edit.guards)
    guardrails.append(_make_rerank_name_guard(rerank_name))
    apply_patch, try_out_patch, revert_changes = make_patch_fn(
        search_fn=primary_search_tool,
        tool_fns=search_tools,
        corpus=corpus,
        code_dir=str(output_dir),
        module_name="reranker",
        function_name=rerank_name,
        guardrail_fns=guardrails,
        training_eval_fn=training_eval_fn,
        validation_eval_fn=validation_eval_fn,
        eval_margin=eval_cfg.eval_margin,
    )

    run_evals, run_reranker = make_eval_tools(
        corpus=corpus,
        judgments=judgments,
        tool_fns=search_tools,
        rerank_name=rerank_name,
        code_path=code_path,
        seed=eval_cfg.training_seed,
        num_queries=len(training_queries_list),
        queries=training_queries_list,
        workers=workers,
    )

    tools = [*search_tools, apply_patch, run_reranker, run_evals, revert_changes]
    if train_config.try_out_patch:
        tools.append(try_out_patch)

    messages: list[str] = []
    round_summaries: list[dict] = []
    round_ndcgs: list[float] = []
    rounds_log_path = output_dir / "rounds.jsonl"
    total_rounds = previous_rounds + train_config.rounds
    for round_idx in range(previous_rounds, total_rounds):
        print(f"Starting training round {round_idx + 1}/{total_rounds}...")
        code = code_path.read_text(encoding="utf-8")
        system_prompt = build_system_prompt(
            train_config.system_prompt,
            dataset=dataset,
            rerank_name=rerank_name,
            search_tool_names=search_tool_names,
            search_tool_docs=search_tool_docs,
            rerank_params=tool_params + ["query"],
            code=code,
        )
        agent = OpenAIAgent(
            tools=tools,
            model="openai/" + train_config.model,
            response_model=FinalMessage,
        )
        inputs = [{"role": "system", "content": system_prompt}]
        resp: FinalMessage | None = agent.loop(inputs=inputs)
        if resp is not None:
            messages.append(resp.message)
        code = code_path.read_text(encoding="utf-8")
        codegen_strategy = CodeGenSearchStrategy(
            corpus,
            search_fn=primary_search_tool,
            tool_fns=search_tools,
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
        round_ndcgs.append(mean_ndcg)
        print(f"Round {round_idx + 1}/{total_rounds} mean NDCG: {mean_ndcg:.4f}")
        round_record = {
            "round": round_idx + 1,
            "short_name": resp.short_name if resp else None,
            "summary": resp.summary if resp else None,
            "message": resp.message if resp else None,
            "mean_ndcg": mean_ndcg,
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
    metadata = {
        "dataset": dataset,
        "strategy_name": strategy_name,
        "rerank_name": rerank_name,
        "model": train_config.model,
        "rounds": total_rounds,
        "rounds_added": train_config.rounds,
        "continued_from": str(continue_path) if continue_from else None,
        "previous_rounds": previous_rounds,
        "search_tools": normalized_tools,
        "messages": messages,
        "round_summaries": round_summaries,
        "round_ndcgs": round_ndcgs,
        "start_with": train_config.start_with,
        "training_seed": eval_cfg.training_seed,
        "validation_seed": eval_cfg.validation_seed,
        "train_query_fraction": eval_cfg.train_query_fraction,
        "validation_query_fraction": eval_cfg.validation_query_fraction,
        "num_training_queries": len(training_queries_list),
        "num_validation_queries": len(validation_queries_list),
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
        search_fn=primary_search_tool,
        tool_fns=search_tools,
    )
