from __future__ import annotations

import json
from pathlib import Path

from cheat_at_search.strategy import SearchStrategy
from pydantic import BaseModel, Field

from exps.agentic.agent import Agent, SearchResults, SUBAGENT_SYSTEM_PROMPT, trace_logger
from exps.mapping import build_doc_id_lookup, doc_ids_to_indices
from exps.run_dirs import slugify
from exps.tools.wands import WANDS_TOP_CATEGORIES, WandsProductCategory


class SelectedCategories(BaseModel):
    """Categories selected for scatter/gather runs."""

    categories: list[WandsProductCategory] = Field(
        description="Selected WANDS categories to search within."
    )


class ScatterGatherWandsStrategy(SearchStrategy):
    _type = "scatter_gather_wands"
    _default_max_loops = 10

    def __init__(
        self,
        corpus,
        workers: int = 1,
        model: str = "gpt-5-mini",
        reasoning: str = "medium",
        agents: dict | None = None,
        plan: list | None = None,
        stop: list | None = None,
        validators: list | None = None,
        max_loops: int | None = None,
        embeddings_device: str | None = None,
        trace_path: Path | None = None,
        dataset: str | None = None,
        subagent_system_prompt: str = SUBAGENT_SYSTEM_PROMPT,
    ):
        if dataset != "wands":
            raise ValueError("scatter_gather_wands only supports the wands dataset.")
        if trace_path is None:
            raise ValueError("scatter_gather_wands requires trace_path to record traces.")
        if not agents or not plan:
            raise ValueError("scatter_gather_wands requires agents and plan configuration.")

        self.trace_path = trace_path
        self.model = model
        self.reasoning = reasoning
        self.embeddings_device = embeddings_device
        self.stop = stop
        self.validators = validators
        self.max_loops = max_loops if max_loops is not None else self._default_max_loops
        self.subagent_system_prompt = subagent_system_prompt
        self.corpus = corpus
        self._lookup = build_doc_id_lookup(corpus)
        self._plan_prompts = self._parse_plan(plan)

        self._select_agent_cfg = agents.get("select")
        self._scatter_agent_cfg = agents.get("scatter")
        self._gather_agent_cfg = agents.get("gather")
        if not self._select_agent_cfg or not self._scatter_agent_cfg or not self._gather_agent_cfg:
            raise ValueError("scatter_gather_wands requires select, scatter, and gather agents.")

        self._validate_agent_tools()

        self._select_agent = Agent(
            corpus=corpus,
            model=model,
            reasoning=reasoning,
            system_prompt=self._select_agent_cfg.get("system_prompt", ""),
            search_tools=self._select_agent_cfg.get("search_tools") or [],
            agents={"select": self._select_agent_cfg},
            plan=[{"select": self._plan_prompts["select"]}],
            stop=stop,
            validators=validators,
            max_loops=self.max_loops,
            embeddings_device=embeddings_device,
            dataset_name="wands",
            subagent_system_prompt=subagent_system_prompt,
            response_model=SelectedCategories,
        )
        self._scatter_agent = Agent(
            corpus=corpus,
            model=model,
            reasoning=reasoning,
            system_prompt=self._scatter_agent_cfg.get("system_prompt", ""),
            search_tools=self._scatter_agent_cfg.get("search_tools") or [],
            agents={"scatter": self._scatter_agent_cfg},
            plan=[{"scatter": self._plan_prompts["scatter"]}],
            stop=stop,
            validators=validators,
            max_loops=self.max_loops,
            embeddings_device=embeddings_device,
            dataset_name="wands",
            subagent_system_prompt=subagent_system_prompt,
            response_model=SearchResults,
        )
        self._gather_agent = Agent(
            corpus=corpus,
            model=model,
            reasoning=reasoning,
            system_prompt=self._gather_agent_cfg.get("system_prompt", ""),
            search_tools=[],
            agents={"gather": self._gather_agent_cfg},
            plan=[{"gather": self._plan_prompts["gather"]}],
            stop=stop,
            validators=validators,
            max_loops=self.max_loops,
            embeddings_device=embeddings_device,
            dataset_name="wands",
            subagent_system_prompt=subagent_system_prompt,
            response_model=SearchResults,
        )
        super().__init__(corpus, workers=workers)

    @classmethod
    def build(
        cls,
        params: dict,
        *,
        corpus,
        workers: int = 1,
        device: str | None = None,
        dataset: str | None = None,
        trace_path: Path | None = None,
        **kwargs,
    ):
        build_params = dict(params)
        if device and "embeddings_device" not in build_params:
            build_params["embeddings_device"] = device
        build_params["dataset"] = dataset
        build_params["trace_path"] = trace_path
        return cls(corpus, workers=workers, **build_params)

    def _parse_plan(self, plan: list) -> dict[str, str]:
        steps = []
        for entry in plan:
            if isinstance(entry, dict) and len(entry) == 1:
                name, prompt = next(iter(entry.items()))
                if not isinstance(prompt, str):
                    raise ValueError("Plan prompts must be strings.")
                steps.append((name, prompt))
                continue
            raise ValueError("Plan entries must be single-key mappings.")
        names = [name for name, _ in steps]
        if names != ["select", "scatter", "gather"]:
            raise ValueError("Plan must include select, scatter, gather in order.")
        return {name: prompt for name, prompt in steps}

    def _validate_agent_tools(self) -> None:
        gather_tools = self._gather_agent_cfg.get("search_tools") or []
        if gather_tools:
            raise ValueError("gather must not define search tools.")
        scatter_tools = self._scatter_agent_cfg.get("search_tools") or []
        for tool in scatter_tools:
            tool_name = tool if isinstance(tool, str) else next(iter(tool))
            if "prefiltered" not in tool_name:
                raise ValueError("scatter tools must be prefiltered.")

    def _select_categories(self, query: str, trace_dir: Path, logger, trace_path: Path) -> list[str]:
        trace_dir.mkdir(parents=True, exist_ok=True)
        response = self._select_agent.run(
            query=query,
            trace_dir=trace_dir,
            logger=logger,
            trace_path=trace_path,
        )
        if not response.output or not getattr(response.output, "categories", None):
            return []
        categories = []
        for category in response.output.categories:
            if category in WANDS_TOP_CATEGORIES:
                categories.append(category)
        trace_path = trace_dir / "summary.json"
        trace_path.write_text(
            json.dumps({"step": "select", "categories": categories}, indent=2) + "\n",
            encoding="utf-8",
        )
        if logger is not None:
            logger.info(
                "scatter_gather_select_results %s",
                {"categories": categories},
            )
        return categories

    def _gather_category_results(
        self,
        *,
        query: str,
        categories: list[str],
        query_dir: Path,
        logger,
        trace_path: Path,
    ) -> dict[str, list[dict[str, str]]]:
        results_by_category: dict[str, list[dict[str, str]]] = {}
        for category in categories:
            scatter_dir = query_dir / "scatter" / slugify(category, fallback="category")
            scatter_dir.mkdir(parents=True, exist_ok=True)
            response = self._scatter_agent.run(
                query=query,
                trace_dir=scatter_dir,
                format_params={"category": category, "query": query},
                logger=logger,
                trace_path=trace_path,
            )
            doc_ids = []
            if isinstance(response.output, list):
                doc_ids = list(response.output)
            detailed = []
            for doc_id in doc_ids:
                detail = {"id": str(doc_id), "title": "", "description": "", "category": category}
                index = self._lookup.get(str(doc_id)) if self._lookup else None
                if index is not None:
                    row = self.corpus.iloc[index]
                    detail["title"] = str(row.get("title", "") or "")
                    detail["description"] = str(row.get("description", "") or "")
                    detail["category"] = str(row.get("category", category) or category)
                detailed.append(detail)
            results_by_category[category] = detailed
            scatter_summary = scatter_dir / "summary.json"
            scatter_summary.write_text(
                json.dumps(
                    {"step": "scatter", "category": category, "doc_ids": doc_ids},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info(
                "scatter_gather_scatter_results %s",
                {"category": category, "doc_ids": doc_ids},
            )
        return results_by_category

    def _gather_final_results(
        self,
        *,
        query: str,
        query_dir: Path,
        results_by_category: dict[str, list[dict[str, str]]],
        logger,
        trace_path: Path,
    ) -> list[str]:
        gather_dir = query_dir / "gather"
        gather_dir.mkdir(parents=True, exist_ok=True)
        gather_response = self._gather_agent.run(
            query=query,
            trace_dir=gather_dir,
            format_params={
                "results_by_category": json.dumps(results_by_category),
                "query": query,
            },
            logger=logger,
            trace_path=trace_path,
        )
        ranked_results: list[str] = []
        if isinstance(gather_response.output, list):
            ranked_results = list(gather_response.output)
        ranked_results = ranked_results[:10]
        gather_summary = gather_dir / "summary.json"
        gather_summary.write_text(
            json.dumps({"step": "gather", "doc_ids": ranked_results}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        logger.info(
            "scatter_gather_gather_results %s",
            {"doc_ids": ranked_results},
        )
        return ranked_results

    def search(self, query: str, k: int = 10):
        query_dir = self.query_path(query)
        with trace_logger(query_dir) as (logger, trace_path):
            select_dir = query_dir / "select"
            categories = self._select_categories(query, select_dir, logger, trace_path)
            if not categories:
                categories = WANDS_TOP_CATEGORIES[:3]
            summary_path = query_dir / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "query": query,
                        "scatter_count": len(categories),
                        "categories": categories,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info(
                "scatter_gather_scatter_count %s",
                {"query": query, "scatter_count": len(categories), "categories": categories},
            )
            results_by_category = self._gather_category_results(
                query=query,
                categories=categories,
                query_dir=query_dir,
                logger=logger,
                trace_path=trace_path,
            )
            ranked_results = self._gather_final_results(
                query=query,
                query_dir=query_dir,
                results_by_category=results_by_category,
                logger=logger,
                trace_path=trace_path,
            )
            if self._lookup:
                ranked_results = doc_ids_to_indices(ranked_results, self._lookup)
            return ranked_results[:k], [1.0] * len(ranked_results[:k])

    def query_path(self, query: str) -> Path:
        if self.trace_path is None:
            raise ValueError("scatter_gather_wands requires trace_path to record traces.")
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
