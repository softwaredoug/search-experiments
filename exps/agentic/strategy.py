from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cheat_at_search.strategy import SearchStrategy
from exps.agentic.agent import (
    Agent,
    DEFAULT_SYSTEM_PROMPT,
    SUBAGENT_SYSTEM_PROMPT,
    _normalize_agents_for_cache,
    _normalize_search_tools_for_cache,
    trace_logger,
)
from exps.agentic.examples import append_few_shot_examples
from exps.mapping import doc_ids_to_indices
from exps.run_dirs import dataset_from_trace_path, slugify
from exps.tools import normalize_search_tools


class AgenticSearchStrategy(SearchStrategy):
    _type = "agentic"
    _default_max_loops = 10

    def __init__(
        self,
        corpus,
        workers: int = 1,
        model: str = "gpt-5-mini",
        reasoning: str = "medium",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        search_tools: list | None = None,
        subagent_system_prompt: str = SUBAGENT_SYSTEM_PROMPT,
        agents: dict | None = None,
        plan: list | None = None,
        stop: list | None = None,
        validators: list | None = None,
        max_loops: int | None = None,
        embeddings_device: str | None = None,
        trace_path: Path | None = None,
    ):
        self.embeddings_device = embeddings_device
        self.trace_path = trace_path
        self.dataset_name = dataset_from_trace_path(trace_path) if trace_path else None
        self.search_tools = search_tools or ["bm25"]
        self.agents = agents
        self.plan = plan
        if self.plan and not self.agents:
            raise ValueError("plan requires agents configuration.")
        self.subagent_system_prompt = subagent_system_prompt
        self.stop = stop
        self.validators = validators
        self.max_loops = max_loops if max_loops is not None else self._default_max_loops
        self.model = model
        self.reasoning = reasoning
        self.system_prompt = system_prompt
        self.corpus = corpus
        self.agent = Agent(
            corpus=corpus,
            model=model,
            reasoning=reasoning,
            system_prompt=system_prompt,
            search_tools=self.search_tools,
            subagent_system_prompt=subagent_system_prompt,
            agents=agents,
            plan=plan,
            stop=stop,
            validators=validators,
            max_loops=self.max_loops,
            embeddings_device=embeddings_device,
            dataset_name=self.dataset_name,
        )
        self.traces: dict[str, str] = {}
        self.num_tool_calls: dict[str, int] = {}
        super().__init__(corpus, workers=workers)

    @classmethod
    def build(
        cls,
        params: dict,
        *,
        corpus,
        workers: int = 1,
        device: str | None = None,
        **kwargs,
    ):
        build_params = dict(params)
        few_shot = build_params.pop("few_shot", None)
        if device and "embeddings_device" not in build_params:
            tool_config = build_params.get("search_tools") or ["bm25"]
            tool_names = [tool["name"] for tool in normalize_search_tools(tool_config)]
            if "minilm" in tool_names:
                build_params["embeddings_device"] = device
        if few_shot:
            system_prompt = build_params.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
            build_params["system_prompt"] = append_few_shot_examples(
                system_prompt,
                corpus=corpus,
                judgments=kwargs.get("judgments"),
                few_shot_config=few_shot,
            )
        return cls(corpus, workers=workers, **build_params)

    def search(self, query: str, k: int = 10):
        if self.trace_path is None:
            raise ValueError("AgenticSearchStrategy requires trace_path to record traces.")
        query_dir = self.query_path(query)
        with trace_logger(query_dir) as (logger, trace_path):
            run_result = self.agent.run(
                query=query,
                trace_dir=query_dir,
                logger=logger,
                trace_path=trace_path,
                k=k,
            )
        ranked_results = run_result.output if isinstance(run_result.output, list) else []
        if self.agent.lookup:
            ranked_results = doc_ids_to_indices(ranked_results, self.agent.lookup)
        self.traces[query] = str(run_result.trace_path)
        num_tool_calls = run_result.num_tool_calls
        self.num_tool_calls[query] = num_tool_calls
        summary_path = query_dir / "summary.json"
        summary_path.write_text(
            json.dumps({"num_tool_calls": num_tool_calls}, indent=2) + "\n",
            encoding="utf-8",
        )
        return ranked_results, [1.0] * len(ranked_results)

    def query_path(self, query: str) -> Path:
        if self.trace_path is None:
            raise ValueError("AgenticSearchStrategy requires trace_path to record traces.")
        query_slug = slugify(query, fallback="query")
        query_dir = self.trace_path / query_slug
        if query_dir.exists():
            counter = 2
            while True:
                candidate = self.trace_path / f"{query_slug}_{counter}"
                try:
                    candidate.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    counter += 1
                    continue
                query_dir = candidate
                break
        return query_dir

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model": self.model,
            "reasoning": self.reasoning,
            "system_prompt": self.system_prompt,
            "search_tools": _normalize_search_tools_for_cache(self.search_tools),
            "subagent_system_prompt": self.subagent_system_prompt,
            "agents": _normalize_agents_for_cache(self.agents),
            "plan": self.plan,
            "stop": self.stop,
            "validators": self.validators,
            "max_loops": self.max_loops,
            "embeddings_device": self.embeddings_device,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
