from __future__ import annotations

from cheat_at_search.strategy import SearchStrategy

from prf.agentic import (
    DEFAULT_SYSTEM_PROMPT,
    SearchResultsGraded,
    agent_run,
    degrade_hook_check,
    grades,
    make_tool_info,
)
from prf.mapping import build_doc_id_lookup, doc_ids_to_indices
from prf.tools import (
    make_bm25_tool,
    make_embedding_tool,
    make_guarded_bm25_tool,
    make_guarded_embedding_tool,
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
            embedding_tool = make_guarded_embedding_tool(
                make_embedding_tool(corpus),
                func_name="search_embeddings_guarded",
            )
            bm25_tool = make_guarded_bm25_tool(
                make_bm25_tool(corpus), func_name="search_bm25_guarded"
            )
            self.tools = [embedding_tool, bm25_tool]
        else:
            self.tools = tools

        super().__init__(corpus, workers=workers)

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
        validator = degrade_hook_check(query)
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
        tool_info = make_tool_info(self.tools)
        resp = None
        while True:
            print("********")
            print(f"ROUND {tries}")
            valid = False
            while not valid:
                resp, inputs = agent_run(
                    tool_info,
                    text_format=SearchResultsGraded,
                    inputs=inputs,
                    model=self.model,
                    agent_state=agent_state,
                )
                valid = validator(resp.output_parsed, inputs)
                if not valid:
                    print("Validation check failed!")

            graded = grades(query, resp.output_parsed)
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

        ranked_results = [r.doc_id for r in resp.output_parsed.ranked_results][:k]
        if self._lookup:
            ranked_results = doc_ids_to_indices(ranked_results, self._lookup)
        return ranked_results, [1.0] * len(ranked_results)
