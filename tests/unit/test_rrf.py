import pytest

from exps.tools.composite import make_rrf_tool


def test_rrf_fuses_weighted_rankings_and_deduplicates_documents():
    calls = []
    agent_state = {"run": "current"}

    def lexical(keywords, top_k=5, agent_state=None):
        calls.append(("lexical", keywords, top_k, agent_state))
        return [
            {"id": "a", "title": "A from lexical", "score": 100.0},
            {"id": "b", "title": "B from lexical", "score": 50.0},
            {"id": "c", "title": "C from lexical", "score": 10.0},
        ][:top_k]

    def semantic(question, top_k=5, agent_state=None):
        calls.append(("semantic", question, top_k, agent_state))
        return [
            {"id": "b", "title": "B from semantic", "score": 0.9},
            {"id": "d", "title": "D from semantic", "score": 0.8},
            {"id": "a", "title": "A from semantic", "score": 0.7},
        ][:top_k]

    search = make_rrf_tool(
        search_tools=[lexical, semantic],
        weights=[2.0, 1.0],
        rank_constant=60,
    )

    results = search("desk chair", top_k=4, agent_state=agent_state)

    assert [result["id"] for result in results] == ["a", "b", "c", "d"]
    assert [result["score"] for result in results] == pytest.approx(
        [2 / 61 + 1 / 63, 2 / 62 + 1 / 61, 2 / 63, 1 / 62]
    )
    assert results[0]["title"] == "A from lexical"
    assert results[1]["title"] == "B from lexical"
    assert calls == [
        ("lexical", "desk chair", 4, agent_state),
        ("semantic", "desk chair", 4, agent_state),
    ]


def test_rrf_requires_one_weight_per_search_tool():
    def first(query, top_k=5, agent_state=None):
        return []

    def second(query, top_k=5, agent_state=None):
        return []

    with pytest.raises(ValueError, match="weights"):
        make_rrf_tool(search_tools=[first, second], weights=[1.0])


def test_rrf_fuses_reranker_candidate_depth_before_reranking():
    arm_calls = []

    def first(query, top_k=5, agent_state=None):
        arm_calls.append(("first", top_k))
        return [{"id": doc_id, "score": 1.0} for doc_id in ["a", "b", "c"][:top_k]]

    def second(query, top_k=5, agent_state=None):
        arm_calls.append(("second", top_k))
        return [{"id": doc_id, "score": 1.0} for doc_id in ["b", "c", "a"][:top_k]]

    class ReverseReranker:
        k = 3

        def rerank(self, *, query, candidates, agent_state=None):
            assert query == "desk"
            assert [candidate["id"] for candidate in candidates] == ["b", "a", "c"]
            return list(reversed(candidates))

    search = make_rrf_tool(
        search_tools=[first, second],
        weights=[1, 1],
        reranker=ReverseReranker(),
    )

    results = search("desk", top_k=1)

    assert arm_calls == [("first", 3), ("second", 3)]
    assert [candidate["id"] for candidate in results] == ["c"]
