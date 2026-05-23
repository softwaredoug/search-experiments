from __future__ import annotations

import hashlib
import json

from cheat_at_search.strategy import SearchStrategy

from cheat_at_search.agent.openai_agent import OpenAIAgent

from exps.agentic.strategy import DEFAULT_SYSTEM_PROMPT, SearchResults
from exps.mapping import build_doc_id_lookup, doc_ids_to_indices
from exps.tools import (
    make_bm25_tool,
    make_embedding_tool,
    make_guarded_search_tool,
)


class AgenticSearchStrategyRalphed(SearchStrategy):
    def __init__(
        self,
        corpus,
        workers: int = 1,
        model: str = "gpt-5",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        skills: dict[str, str] | None = None,
        tools=None,
    ):
        self.model = model
        self.skills = skills or {}
        self.system_prompt = system_prompt
        self._lookup = build_doc_id_lookup(corpus)

        if tools is None:
            embedding_tool = make_guarded_search_tool(
                make_embedding_tool(corpus),
                func_name="search_minilm_guarded",
            )
            bm25_tool = make_guarded_search_tool(
                make_bm25_tool(corpus), func_name="search_bm25_guarded"
            )
            self.tools = [embedding_tool, bm25_tool]
        else:
            self.tools = tools

        super().__init__(corpus, workers=workers)

    @property
    def cache_key(self) -> str:
        payload = {
            "type": "agentic_ralphed",
            "model": self.model,
            "system_prompt": self.system_prompt,
            "skills": self.skills,
            "tools": ["minilm", "bm25"],
            "guards": [],
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()

    def _inject_skill_on_kw(self, query: str) -> list[str]:
        skill_prompts = []
        for skill_rule, skill_prompt in self.skills.items():
            if skill_rule in query:
                skill_prompts.append(skill_prompt)
        return skill_prompts

    def _title_for_doc_id(self, doc_id: int) -> str:
        if "doc_id" not in self.corpus.columns:
            return ""
        match = self.corpus[self.corpus["doc_id"] == doc_id]
        if match.empty:
            return ""
        title = match.iloc[0].get("title", "")
        return str(title) if title is not None else ""

    def search(self, query: str, k: int = 10):
        validator = _degrade_hook_check(query)
        agentic_query = "Find me: " + query
        inputs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": agentic_query},
        ]

        for prompt in self._inject_skill_on_kw(query):
            print(f"Using skill: {prompt}")
            inputs.append({"role": "user", "content": prompt})

        tries = 0
        agent_state = {"past_queries": {}}
        agent = OpenAIAgent(
            tools=self.tools,
            model=f"openai/{self.model}" if "/" not in self.model else self.model,
            response_model=SearchResults,
            reasoning_level="medium",
        )
        resp = None
        while True:
            print("********")
            print(f"ROUND {tries}")
            valid = False
            while not valid:
                resp = agent.loop(inputs=inputs, agent_state=agent_state)
                valid = validator(resp, inputs)
                if not valid:
                    print("Validation check failed!")

            graded = _grades(query, resp)
            message_back = (
                "These results can be improved. Can you look at them and fix them?\n\n"
                "Get creative\n\n"
                "Here's how I feel about the results, please improve them.\n"
                "Try to find more 😃 results to replace the 😑 ones.\n"
                "Or at least 😑 to replace ☹️\n\n"
                "Return a better ranking, with the happier emojis towards the top\n\n"
                f"Reminder the search is for: {query}\n\n"
            )
            for doc_id, emoji in graded:
                title = self._title_for_doc_id(doc_id)
                label = title if title else str(doc_id)
                message_back += f"{emoji} {label}\n"
            print(message_back)
            inputs.append({"role": "user", "content": message_back})
            tries += 1
            if tries > 3:
                break

        ranked_results = [r.doc_id for r in resp.ranked_results][:k]
        if self._lookup:
            ranked_results = doc_ids_to_indices(ranked_results, self._lookup)
        return ranked_results, [1.0] * len(ranked_results)


def _grade_to_emoji(grade):
    if grade == 0:
        return "☹️"
    if grade == 1:
        return "😑"
    if grade == 2:
        return "😃"
    return "☹️"


def _grades(query: str, search_results: SearchResults):
    from cheat_at_search.wands_data import labeled_query_products

    query_judgments = labeled_query_products[labeled_query_products["query"] == query]
    results = []
    ranked_results = search_results.ranked_results or []
    for doc_id in ranked_results:
        try:
            doc_id = int(doc_id)
        except (TypeError, ValueError):
            continue
        doc_judgments = query_judgments[query_judgments["doc_id"] == doc_id]
        if len(doc_judgments) == 0:
            results.append((doc_id, _grade_to_emoji(None)))
        else:
            grade = int(doc_judgments["grade"].values[0])
            results.append((doc_id, _grade_to_emoji(grade)))
    return results


def _count_smileys(gradeds):
    count = 0
    for graded in gradeds:
        if graded[1] == "😃":
            count += 1
    return count


def _degrade_hook_check(query: str):
    def search_degrade_hook(resp, inputs):
        all_graded = []
        for input_item in inputs:
            if hasattr(input_item, "content") and input_item.content is not None:
                content = input_item.content
                if content and hasattr(content[-1], "parsed"):
                    result = content[-1].parsed
                    if isinstance(result, SearchResults):
                        all_graded.append(_grades(query, result))
        if len(all_graded) > 1:
            last_smileys = _count_smileys(all_graded[-2])
            current_smileys = _count_smileys(all_graded[-1])
            if last_smileys > current_smileys:
                inputs.append(
                    {
                        "role": "user",
                        "content": (
                            "Oh this isn't good, it turns out: You've degraded your relevance, "
                            f"previously found {last_smileys} relevant results , and now found "
                            f"{current_smileys}. Please try again"
                        ),
                    }
                )
                return False
        return True

    return search_degrade_hook
