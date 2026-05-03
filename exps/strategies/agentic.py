from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from cheat_at_search.strategy import SearchStrategy

from exps.agentic import (
    DEFAULT_SYSTEM_PROMPT,
    SearchResultsIds,
    normalize_stops_for_cache,
    search,
    trace_logger,
)
from exps.mapping import build_doc_id_lookup, doc_ids_to_indices
from exps.trace_utils import dataset_from_trace_path, slugify
from exps.tools import (
    build_search_tools,
    normalize_search_tools,
    normalize_search_tools_for_cache,
)


def _grade_column(judgments):
    for col in ("grade", "relevance", "rel", "label", "score"):
        if col in judgments.columns:
            return col
    return None


def _grade_to_emoji(grade, grade_levels):
    if not grade_levels:
        return "😐"
    if len(grade_levels) == 1:
        return "😐"
    if grade == grade_levels[0]:
        return "😭"
    if grade == grade_levels[-1]:
        return "😃"
    return "😐"


def _sorted_grades(values):
    def _coerce(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    numeric = [value for value in values if _coerce(value) is not None]
    if len(numeric) == len(values):
        return sorted(numeric)
    return sorted(values, key=lambda value: str(value))


def _append_few_shot_examples(
    system_prompt: str,
    *,
    corpus,
    judgments,
    few_shot_config: list,
) -> str:
    if judgments is None:
        raise ValueError("few_shot requires judgments to be available.")
    grade_col = _grade_column(judgments)
    if grade_col is None:
        raise ValueError("few_shot requires a grade column in judgments.")
    if "query" not in judgments.columns or "doc_id" not in judgments.columns:
        raise ValueError("few_shot requires query and doc_id columns in judgments.")

    corpus_lookup = None
    if "doc_id" in corpus.columns:
        corpus_lookup = corpus.set_index("doc_id", drop=False)

    blocks = []
    for entry in few_shot_config:
        if isinstance(entry, dict) and "sample_judgments" in entry:
            raw = entry["sample_judgments"]
            if isinstance(raw, dict):
                if "num_rows" not in raw:
                    raise ValueError("few_shot.sample_judgments requires num_rows.")
                sample_count = int(raw["num_rows"])
                columns = raw.get("columns") or []
            else:
                sample_count = int(raw)
                columns = []
            if sample_count <= 0:
                continue
            if not isinstance(columns, list):
                raise ValueError("few_shot.sample_judgments.columns must be a list.")
            for col in columns:
                if col not in corpus.columns:
                    raise ValueError(f"few_shot column not found in corpus: {col}")
            seed = int(entry.get("seed", 42))

            pool = judgments.dropna(subset=[grade_col, "query", "doc_id"])
            grades = list(pool[grade_col].dropna().unique())
            grades = _sorted_grades(grades)
            if not grades:
                continue
            rng = random.Random(seed)
            grouped = {
                grade: pool[pool[grade_col] == grade].sample(frac=1.0, random_state=rng.randrange(1 << 30))
                for grade in grades
            }
            queues = {grade: grouped[grade].iterrows() for grade in grades}
            samples = []
            while len(samples) < sample_count:
                advanced = False
                for grade in grades:
                    try:
                        _, row = next(queues[grade])
                    except StopIteration:
                        continue
                    samples.append(row)
                    advanced = True
                    if len(samples) >= sample_count:
                        break
                if not advanced:
                    break

            lines = [
                "Few-shot examples (query, product, relevance):",
            ]
            for row in samples:
                query = row.get("query")
                doc_id = row.get("doc_id")
                grade = row.get(grade_col)
                emoji = _grade_to_emoji(grade, grades)
                title = ""
                description = ""
                extra_fields = {}
                if corpus_lookup is not None and doc_id in corpus_lookup.index:
                    match = corpus_lookup.loc[doc_id]
                    if hasattr(match, "ndim") and match.ndim > 1:
                        match = match.iloc[0]
                    if hasattr(match, "get"):
                        title = match.get("title", "")
                        description = match.get("description", "")
                        for col in columns:
                            extra_fields[col] = match.get(col, "")
                lines.extend(
                    [
                        f"Query: {query}",
                        f"Doc ID: {doc_id}",
                        f"Title: {title}",
                        f"Description: {description}",
                    ]
                )
                for col in columns:
                    lines.append(f"{col}: {extra_fields.get(col, '')}")
                lines.extend(
                    [
                        f"Relevance: {emoji}",
                        "",
                    ]
                )
            blocks.append("\n".join(lines).strip())
            continue
        raise ValueError("few_shot entries must be mappings with sample_judgments.")

    if not blocks:
        return system_prompt
    return system_prompt.rstrip() + "\n\n" + "\n\n".join(blocks) + "\n"


class AgenticSearchStrategy(SearchStrategy):
    _type = "agentic"

    def __init__(
        self,
        corpus,
        workers: int = 1,
        model: str = "gpt-5-mini",
        reasoning: str = "medium",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        search_tools: list | None = None,
        stop: list | None = None,
        reprompt: str | None = None,
        embeddings_device: str | None = None,
        trace_path: Path | None = None,
    ):
        self.embeddings_device = embeddings_device
        self.trace_path = trace_path
        dataset_name = dataset_from_trace_path(trace_path) if trace_path else None
        tool_config = search_tools or ["bm25"]
        self.search_tools = tool_config
        self.stop = stop
        self.reprompt = reprompt
        self.tools = build_search_tools(
            corpus,
            tool_config,
            embeddings_device=embeddings_device,
            dataset_name=dataset_name,
        )
        self.model = model
        self.reasoning = reasoning
        self.system_prompt = system_prompt
        self._lookup = build_doc_id_lookup(corpus)
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
            build_params["system_prompt"] = _append_few_shot_examples(
                system_prompt,
                corpus=corpus,
                judgments=kwargs.get("judgments"),
                few_shot_config=few_shot,
            )
        return cls(corpus, workers=workers, **build_params)

    def search(self, query: str, k: int = 10):
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
        inputs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        agent_state = {"num_tool_calls": 0}
        with trace_logger(query_dir) as (logger, trace_path):
            logger.info("Query: %s", query)
            resp = search(
                tools=self.tools,
                inputs=inputs,
                agent_state=agent_state,
                model=self.model,
                reasoning=self.reasoning,
                text_format=SearchResultsIds,
                logger=logger,
                stop=self.stop,
                reprompt=self.reprompt,
            )
            ranked_results = resp.ranked_results[:k]
            if self._lookup:
                ranked_results = doc_ids_to_indices(ranked_results, self._lookup)
        self.traces[query] = str(trace_path)
        num_tool_calls = int(agent_state.get("num_tool_calls", 0))
        self.num_tool_calls[query] = num_tool_calls
        summary_path = query_dir / "summary.json"
        summary_path.write_text(
            json.dumps({"num_tool_calls": num_tool_calls}, indent=2) + "\n",
            encoding="utf-8",
        )
        return ranked_results, [1.0] * len(ranked_results)

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model": self.model,
            "reasoning": self.reasoning,
            "system_prompt": self.system_prompt,
            "search_tools": normalize_search_tools_for_cache(self.search_tools),
            "stop": normalize_stops_for_cache(self.stop),
            "reprompt": self.reprompt,
            "embeddings_device": self.embeddings_device,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
