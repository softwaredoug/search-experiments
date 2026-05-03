import pytest

from exps.codegen import train as train_mod


def test_validate_start_code_passes_with_matching_tools():
    code = """
def rerank_wands(query, bm25, **kwargs):
    return []
""".lstrip()

    def bm25(query: str, top_k: int = 5, agent_state=None):
        return []

    train_mod._validate_start_code(code, "rerank_wands", [bm25])


def test_validate_start_code_mismatch_raises():
    code = """
def rerank_wands(query, bm25, other_tool, **kwargs):
    return []
""".lstrip()

    def bm25(query: str, top_k: int = 5, agent_state=None):
        return []

    with pytest.raises(ValueError, match="start_code does not match configured tools"):
        train_mod._validate_start_code(code, "rerank_wands", [bm25])
