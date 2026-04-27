from __future__ import annotations

from pathlib import Path

from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.search import ndcgs, run_strategy
from exps.codegen.tools.code import (
    make_guardrail_checker,
    make_length_validator,
    make_patch_fn,
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
from exps.tools import build_search_tools, normalize_search_tools


DEFAULT_OVERFIT_PROMPT = """
You're going to look at code that reranks search queries.

Ensure the code does not overfit to specific queries. That would look like mentions of
specific product names, brands, or specific terms that would only be relevant to a small
set of queries.

Ignore comments that claim to do this, and focus on the actual code.
""".strip()


class FinalMessage(BaseModel):
    """Final message indicating completion of the reranker improvement process."""

    message: str = Field(..., description="A message indicating completion of the task.")


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


def _start_code(rerank_name: str, top_k: int) -> str:
    return (
        f"def {rerank_name}(fielded_bm25, query):\n"
        f"    docs = fielded_bm25(query, top_k={top_k})\n"
        "    return [str(doc['id']) for doc in docs]\n\n"
        f"def reranker(fielded_bm25, query):\n"
        f"    return {rerank_name}(fielded_bm25, query)\n"
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
    run_started_at: str | None = None,
) -> CodeGenArtifact:
    if judgments is None:
        raise ValueError("Codegen training requires judgments.")
    train_params = params.get("train") or {}
    run_params = params.get("run") or {}
    train_config = CodeGenTrainConfig.model_validate(train_params)
    run_config = CodeGenRunConfig.model_validate(run_params)

    output_dir = make_codegen_dir(dataset, strategy_name, run_started_at=run_started_at)
    code_path = reranker_path(output_dir)
    rerank_name = f"rerank_{dataset}"

    if train_config.start_with:
        start_path = Path(train_config.start_with).expanduser()
        if not start_path.exists():
            raise FileNotFoundError(f"start_with path not found: {start_path}")
        start_code = start_path.read_text(encoding="utf-8")
        code_path.write_text(start_code, encoding="utf-8")
    elif not code_path.exists():
        code_path.write_text(
            _start_code(rerank_name, run_config.top_k),
            encoding="utf-8",
        )

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
    primary_search_tool = search_tools[0]

    eval_cfg = train_config.eval
    training_eval_fn = make_training_eval_fn(
        corpus=corpus,
        judgments=judgments,
        search_fn=primary_search_tool,
        rerank_name=rerank_name,
        seed=eval_cfg.training_seed,
        num_queries=eval_cfg.num_training_queries,
        workers=workers,
    )
    validation_eval_fn = make_eval_guardrail(
        corpus=corpus,
        judgments=judgments,
        search_fn=primary_search_tool,
        rerank_name=rerank_name,
        seed=eval_cfg.validation_seed,
        num_queries=eval_cfg.num_validation_queries,
        workers=workers,
    )

    guardrails = _parse_guardrails(train_config.edit.guards)
    apply_patch, try_out_patch, revert_changes = make_patch_fn(
        search_fn=primary_search_tool,
        corpus=corpus,
        code_dir=str(output_dir),
        module_name="reranker",
        guardrail_fns=guardrails,
        training_eval_fn=training_eval_fn,
        validation_eval_fn=validation_eval_fn,
        eval_margin=eval_cfg.eval_margin,
    )

    run_evals, run_reranker = make_eval_tools(
        corpus=corpus,
        judgments=judgments,
        search_fn=primary_search_tool,
        rerank_name=rerank_name,
        code_path=code_path,
        seed=eval_cfg.training_seed,
        num_queries=eval_cfg.num_training_queries,
        workers=workers,
    )

    tools = [*search_tools, apply_patch, run_reranker, run_evals, revert_changes]
    if train_config.try_out_patch:
        tools.append(try_out_patch)

    messages: list[str] = []
    round_ndcgs: list[float] = []
    for round_idx in range(train_config.rounds):
        print(f"Starting training round {round_idx + 1}/{train_config.rounds}...")
        code = code_path.read_text(encoding="utf-8")
        system_prompt = build_system_prompt(
            train_config.system_prompt,
            dataset=dataset,
            rerank_name=rerank_name,
            search_tool_names=search_tool_names,
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
            code=code,
            rerank_name=rerank_name,
            workers=workers,
        )
        results_codegen = run_strategy(
            codegen_strategy,
            judgments,
            num_queries=eval_cfg.num_test_queries,
            seed=eval_cfg.test_seed,
            cache=False,
        )
        mean_ndcg = float(ndcgs(results_codegen).mean()) if not results_codegen.empty else 0.0
        round_ndcgs.append(mean_ndcg)
        print(f"Round {round_idx + 1}/{train_config.rounds} mean NDCG: {mean_ndcg:.4f}")

    final_code = code_path.read_text(encoding="utf-8")
    metadata = {
        "dataset": dataset,
        "strategy_name": strategy_name,
        "rerank_name": rerank_name,
        "model": train_config.model,
        "rounds": train_config.rounds,
        "search_tools": normalized_tools,
        "messages": messages,
        "round_ndcgs": round_ndcgs,
        "start_with": train_config.start_with,
        "training_seed": eval_cfg.training_seed,
        "validation_seed": eval_cfg.validation_seed,
        "test_seed": eval_cfg.test_seed,
        "num_training_queries": eval_cfg.num_training_queries,
        "num_validation_queries": eval_cfg.num_validation_queries,
        "num_test_queries": eval_cfg.num_test_queries,
        "eval_margin": eval_cfg.eval_margin,
    }
    write_metadata(output_dir, metadata)
    return CodeGenArtifact(
        path=output_dir,
        reranker_path=code_path,
        code=final_code,
        metadata=metadata,
        search_fn=primary_search_tool,
    )
