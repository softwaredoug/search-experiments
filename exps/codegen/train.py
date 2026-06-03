from __future__ import annotations

import random
import textwrap
from pathlib import Path

from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.codegen import Reranker
from cheat_at_search.search import ndcgs, run_strategy
from pydantic import BaseModel, Field

from exps.codegen.io import reranker_path, write_metadata
from exps.codegen.prompts import build_system_prompt
from exps.codegen.queries import build_round_queries
from exps.codegen.results import Rounds
from exps.codegen.strategy import CodeGenSearchStrategy
from exps.codegen.types import CodeGenArtifact, CodeGenRunConfig, CodeGenTrainConfig
from exps.codegen.guardrails import GuardrailsConfig, make_rerank_name_guard, parse_guardrails
from exps.codegen.search_tools import (
    SearchToolState,
    _validate_start_code,
    build_search_tool_state,
    normalize_tool_config,
)
from exps.codegen.utils import split_search_tools
from exps.logging_utils import log_to_path_and_stdout

__all__ = [
    "_validate_start_code",
]


class FinalMessage(BaseModel):
    """Final message indicating completion of the reranker improvement process."""

    message: str = Field(..., description="A message indicating completion of the task.")
    short_name: str | None = Field(
        None, description="Short 3-4 word name of the change."
    )
    summary: str | None = Field(
        None, description="One or more sentences describing the change."
    )


def _ensure_reranker_wrapper(code: str, rerank_name: str) -> str:
    exec_globals: dict = {}
    try:
        exec(code, exec_globals)
    except Exception:
        return code
    if rerank_name in exec_globals and callable(exec_globals[rerank_name]):
        return code
    candidate = None
    for name, obj in exec_globals.items():
        if name.startswith("rerank_") and callable(obj):
            candidate = name
            break
    if not candidate:
        return code
    wrapper = (
        f"\n\n"
        f"def {rerank_name}(query, top_k, *tool_fns, **kwargs):\n"
        f"    try:\n"
        f"        return {candidate}(query, top_k=top_k, *tool_fns, **kwargs)\n"
        f"    except TypeError:\n"
        f"        return {candidate}(query, *tool_fns, **kwargs)\n"
    )
    return code.rstrip() + wrapper


class Codegen:
    def __init__(
        self,
        *,
        dataset: str,
        corpus,
        judgments,
        train_config: CodeGenTrainConfig,
        run_config: CodeGenRunConfig,
        run_path: str | Path | None = None,
        device: str | None = None,
        workers: int = 1,
        report_num_queries: int | None = None,
        report_seed: int = 42,
    ) -> None:
        if judgments is None:
            raise ValueError("Codegen training requires judgments.")
        self.dataset = dataset
        self.corpus = corpus
        self.judgments = judgments
        self.train_config = train_config
        self.run_config = run_config
        self.device = device
        self.workers = workers
        self.report_num_queries = report_num_queries
        self.report_seed = report_seed

        self.output_dir: Path | None = None
        self.rerank_name = "reranker"

        tool_config = self.train_config.search_tools or ["bm25"]
        self.normal_tool_config, self.raw_tool_config = split_search_tools(tool_config)
        self.normalized_tools = normalize_tool_config(self.normal_tool_config)

        self.start_code_from_config = False
        self.previous_rounds = 0
        self.start_code: str | None = None
        self.continue_from: str | None = None
        if run_path is None:
            raise ValueError("Codegen training requires a run_path.")
        self.output_dir = Path(run_path).expanduser()
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Codegen run path not found: {self.output_dir}")
        if not self.output_dir.is_dir():
            raise ValueError(f"Codegen run path must be a directory: {self.output_dir}")
        self.previous_rounds, self.start_code = _resolve_path_continuation(self.output_dir)
        self.continue_from = str(self.output_dir) if self.previous_rounds > 0 else None
        if self.start_code is None and self.train_config.start_code:
            self.start_code = textwrap.dedent(self.train_config.start_code).lstrip()
            self.start_code_from_config = True
        if self.start_code is not None:
            normalized = _ensure_reranker_wrapper(self.start_code, self.rerank_name)
            self.code_path.write_text(normalized, encoding="utf-8")

        self.search_tool_state: SearchToolState = build_search_tool_state(
            corpus=corpus,
            dataset=dataset,
            device=device,
            normal_tool_config=self.normal_tool_config,
            raw_tool_config=self.raw_tool_config,
            rerank_name=self.rerank_name,
            code_path=self.code_path,
            start_code_from_config=self.start_code_from_config,
        )
        if not self.code_path.exists():
            self.code_path.write_text(
                _start_code(
                    self.rerank_name,
                    self.run_config.top_k,
                    tool_params=self.search_tool_state.tool_params,
                    primary_tool_name=self.search_tool_state.primary_tool.name,
                ),
                encoding="utf-8",
            )

        log_path = self.output_dir / "codegen.log"
        self.train_logger = log_to_path_and_stdout("codegen.train", log_path)
        self.code_logger = log_to_path_and_stdout("code", log_path)
        log_to_path_and_stdout("eval", log_path)

        self.eval_cfg = self.train_config.eval
        query_cols = ["query"]
        if "query_id" in judgments.columns:
            query_cols.append("query_id")
        available_queries = judgments[query_cols].drop_duplicates()["query"].tolist()
        if report_num_queries is not None:
            report_count = min(report_num_queries, len(available_queries))
            self.base_queries = random.Random(report_seed).sample(available_queries, report_count)
        else:
            self.base_queries = list(available_queries)
            report_count = len(self.base_queries)
        self.report_count = report_count

        train_size = int(len(self.base_queries) * self.eval_cfg.train_fraction)
        if self.base_queries and self.eval_cfg.train_fraction > 0 and train_size == 0:
            train_size = 1
        self.train_size = train_size

        guardrails_config: GuardrailsConfig = parse_guardrails(
            self.train_config.edit.guards,
            logger=self.code_logger,
        )
        self.guardrails = guardrails_config.guardrails
        self.validation_enabled = guardrails_config.validation_enabled
        self.guardrails.append(make_rerank_name_guard(self.rerank_name))

        self.rounds_log_path = self.output_dir / "rounds.jsonl"
        self.rounds = Rounds(self.rounds_log_path)

        self.refresh_every = self.train_config.refresh_every or self.train_config.rounds
        if self.refresh_every <= 0:
            raise ValueError("refresh_every must be >= 1")

        self.training_queries_list: list[str] = []
        self.validation_queries_list: list[str] = []
        self.test_queries_list: list[str] = []
        self.tools: list[callable] = []
        self.reranker: Reranker | None = None
        self.total_rounds = self.previous_rounds + self.train_config.rounds
        self.current_round = self.previous_rounds

    @property
    def code_path(self) -> Path:
        if self.output_dir is None:
            raise ValueError("Codegen output_dir is not initialized.")
        return reranker_path(self.output_dir)

    def _build_tools(self) -> list[callable]:
        validation_queries = (
            self.validation_queries_list if self.validation_enabled else None
        )
        tool_fns = [tool.fn for tool in self.search_tool_state.tools]
        self.reranker = Reranker(
            code_dir=str(self.code_path.parent),
            tool_fns=tool_fns,
            corpus=self.corpus,
            judgments=self.judgments,
            training_queries=self.training_queries_list,
            validation_queries=validation_queries,
            module_name=self.rerank_name,
            guardrail_fns=self.guardrails,
            eval_margin=self.eval_cfg.eval_margin,
            logger=self.code_logger,
            files={
                "rounds.jsonl": "per-round summaries",
                "codegen.log": "training log",
                "metadata.json": "run metadata",
                "reranker_round_*.py": "per-round reranker snapshots",
            },
        )
        search_tool, evaluate_tool, commit_patch_tool, grep_tool = self.reranker.tools()

        return [
            *[tool.fn for tool in self.search_tool_state.search_tools],
            search_tool,
            evaluate_tool,
            commit_patch_tool,
            grep_tool,
        ]

    def _refresh_round_state(self, round_idx: int) -> None:
        refresh_round = (round_idx - self.previous_rounds) % self.refresh_every == 0
        if not self.tools or refresh_round:
            (
                self.training_queries_list,
                self.validation_queries_list,
                self.test_queries_list,
            ) = build_round_queries(
                round_idx=round_idx,
                base_queries=self.base_queries,
                train_size=self.train_size,
                eval_cfg=self.eval_cfg,
            )
            self.tools = self._build_tools()
            self.start_code_from_config = False

    def initialize_baseline(self) -> None:
        if self.previous_rounds != 0 or self.rounds_log_path.exists():
            return
        self._refresh_round_state(self.previous_rounds)
        baseline_code = self.code_path.read_text(encoding="utf-8")
        baseline_strategy = CodeGenSearchStrategy(
            self.corpus,
            search_fn=self.search_tool_state.primary_tool.fn,
            tool_fns=[tool.fn for tool in self.search_tool_state.tools],
            code=baseline_code,
            rerank_name=self.rerank_name,
            workers=self.workers,
        )
        baseline_results = run_strategy(
            baseline_strategy,
            self.judgments,
            queries=self.base_queries,
            seed=self.report_seed,
            cache=False,
        )
        baseline_ndcg = float(ndcgs(baseline_results).mean()) if not baseline_results.empty else 0.0
        baseline_test_ndcg = 0.0
        if self.test_queries_list:
            baseline_test_results = run_strategy(
                baseline_strategy,
                self.judgments,
                queries=self.test_queries_list,
                seed=self.report_seed,
                cache=False,
            )
            baseline_test_ndcg = (
                float(ndcgs(baseline_test_results).mean())
                if not baseline_test_results.empty
                else 0.0
            )
        self.rounds.finish(
            round_number=0,
            short_name="baseline",
            summary="Initial reranker baseline",
            message=None,
            mean_ndcg=baseline_ndcg,
            mean_test_ndcg=baseline_test_ndcg,
            training_query_count=len(self.training_queries_list),
            validation_query_count=(
                len(self.validation_queries_list) if self.validation_enabled else 0
            ),
            test_query_count=len(self.test_queries_list),
        )
        baseline_code_path = self.output_dir / "reranker_round_0.py"
        baseline_code_path.write_text(baseline_code, encoding="utf-8")

    def train(self) -> dict:
        if self.current_round >= self.total_rounds:
            raise ValueError("No remaining rounds to train.")
        self._refresh_round_state(self.current_round)

        self.train_logger.info(
            "Starting training round %s/%s...",
            self.current_round + 1,
            self.total_rounds,
        )
        code = self.code_path.read_text(encoding="utf-8")
        system_prompt = build_system_prompt(
            self.train_config.system_prompt,
            dataset=self.dataset,
            rerank_name=self.rerank_name,
            search_tool_names=[tool.name for tool in self.search_tool_state.search_tools],
            search_tool_docs=[tool.doc for tool in self.search_tool_state.search_tools],
            raw_tool_names=[tool.name for tool in self.search_tool_state.raw_tools],
            raw_tool_docs=[tool.doc for tool in self.search_tool_state.raw_tools],
            rerank_params=["query", "top_k", *self.search_tool_state.tool_params, "**kwargs"],
            code=code,
        )
        agent = OpenAIAgent(
            tools=self.tools,
            model="openai/" + self.train_config.model,
            response_model=FinalMessage,
            reasoning_level=self.train_config.reasoning,
        )
        inputs = [{"role": "system", "content": system_prompt}]
        resp: FinalMessage | None = agent.loop(inputs=inputs)
        message = resp.message if resp else None
        code = self.code_path.read_text(encoding="utf-8")
        codegen_strategy = CodeGenSearchStrategy(
            self.corpus,
            search_fn=self.search_tool_state.primary_tool.fn,
            tool_fns=[tool.fn for tool in self.search_tool_state.tools],
            code=code,
            rerank_name=self.rerank_name,
            workers=self.workers,
        )
        results_codegen = run_strategy(
            codegen_strategy,
            self.judgments,
            queries=self.base_queries,
            seed=self.report_seed,
            cache=False,
        )
        mean_ndcg = float(ndcgs(results_codegen).mean()) if not results_codegen.empty else 0.0
        mean_test_ndcg = 0.0
        if self.test_queries_list:
            results_test = run_strategy(
                codegen_strategy,
                self.judgments,
                queries=self.test_queries_list,
                seed=self.report_seed,
                cache=False,
            )
            mean_test_ndcg = float(ndcgs(results_test).mean()) if not results_test.empty else 0.0
        self.train_logger.info(
            "Round %s/%s mean NDCG: %.4f (test %.4f)",
            self.current_round + 1,
            self.total_rounds,
            mean_ndcg,
            mean_test_ndcg,
        )
        round_record = self.rounds.finish(
            round_number=self.current_round + 1,
            short_name=resp.short_name if resp else None,
            summary=resp.summary if resp else None,
            message=message,
            mean_ndcg=mean_ndcg,
            mean_test_ndcg=mean_test_ndcg,
            training_query_count=len(self.training_queries_list),
            validation_query_count=(
                len(self.validation_queries_list) if self.validation_enabled else 0
            ),
            test_query_count=len(self.test_queries_list),
        )
        round_code_path = self.output_dir / f"reranker_round_{self.current_round + 1}.py"
        round_code_path.write_text(code, encoding="utf-8")
        self.current_round += 1
        return round_record

    def run(self, queries: list[str]):
        self._refresh_round_state(self.current_round)
        strategy = CodeGenSearchStrategy(
            self.corpus,
            search_fn=self.search_tool_state.primary_tool.fn,
            tool_fns=[tool.fn for tool in self.search_tool_state.tools],
            code=self.code_path.read_text(encoding="utf-8"),
            rerank_name=self.rerank_name,
            workers=self.workers,
        )
        return run_strategy(
            strategy,
            self.judgments,
            queries=queries,
            seed=self.report_seed,
            cache=False,
        )


def _start_code(
    rerank_name: str,
    top_k: int,
    *,
    tool_params: list[str],
    primary_tool_name: str,
) -> str:
    signature = ", ".join(["query", "top_k", *tool_params, "**kwargs"])
    if "fielded_bm25" in primary_tool_name:
        call = (
            f"    docs = {primary_tool_name}"
            f"(query, fields=['title^9.3', 'description^4.1'], operator='or', top_k=top_k)\n"
        )
    elif primary_tool_name == "get_corpus":
        call = (
            "    corpus = get_corpus()\n"
            "    docs = corpus.head(top_k).to_dict('records')\n"
        )
    else:
        call = f"    docs = {primary_tool_name}(query, top_k=top_k)\n"
    return (
        f"def {rerank_name}({signature}):\n"
        f"{call}"
        "    return [str(doc['id']) for doc in docs]\n\n"
        "def rerank_default(query, top_k, *tool_fns, **kwargs):\n"
        f"    return {rerank_name}(query, top_k, *tool_fns, **kwargs)\n"
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
) -> CodeGenArtifact:
    if run_path is None:
        raise ValueError("Codegen training requires a run_path; set strategy.path in the config.")
    train_params = params.get("train") or {}
    run_params = params.get("run") or {}
    if "path" in run_params:
        raise ValueError("run.path is no longer supported; use strategy.path instead.")
    train_config = CodeGenTrainConfig.model_validate(train_params)
    run_config = CodeGenRunConfig.model_validate(run_params)
    if train_config.continue_from or "continue" in train_params:
        raise ValueError("train.continue is no longer supported; use strategy.path instead.")
    if train_params.get("start_with"):
        raise ValueError("start_with is no longer supported; use strategy.path instead.")

    codegen = Codegen(
        dataset=dataset,
        corpus=corpus,
        judgments=judgments,
        train_config=train_config,
        run_config=run_config,
        run_path=run_path,
        device=device,
        workers=workers,
        report_num_queries=report_num_queries,
        report_seed=report_seed,
    )
    codegen.initialize_baseline()
    while codegen.current_round < codegen.total_rounds:
        codegen.train()

    final_code = codegen.code_path.read_text(encoding="utf-8")
    metadata = {
        "dataset": dataset,
        "strategy_name": strategy_name,
        "rerank_name": codegen.rerank_name,
        "model": train_config.model,
        "rounds": codegen.total_rounds,
        "rounds_added": train_config.rounds,
        "continued_from": codegen.continue_from,
        "previous_rounds": codegen.previous_rounds,
        "refresh_every": codegen.refresh_every,
        "search_tools": codegen.normalized_tools,
        "seed": codegen.eval_cfg.seed,
        "train_fraction": codegen.eval_cfg.train_fraction,
        "num_training_queries": len(codegen.training_queries_list),
        "num_validation_queries": (
            len(codegen.validation_queries_list) if codegen.validation_enabled else 0
        ),
        "base_query_count": len(codegen.base_queries),
        "report_seed": report_seed,
        "report_num_queries": report_num_queries,
        "eval_margin": codegen.eval_cfg.eval_margin,
    }
    write_metadata(codegen.output_dir, metadata)
    rval = CodeGenArtifact(
        path=codegen.output_dir,
        reranker_path=codegen.code_path,
        code=final_code,
        metadata=metadata,
        rounds=codegen.rounds,
        search_fn=codegen.search_tool_state.primary_tool.fn,
        tool_fns=[tool.fn for tool in codegen.search_tool_state.tools],
    )
    return rval
