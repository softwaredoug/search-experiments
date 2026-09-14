from __future__ import annotations

from unittest.mock import patch

from exps.runners.run import RunParams, run_benchmark


class _FakeQueryAgent:
    last_inputs = None

    def __init__(self, tools, model, response_model, reasoning_level, images=False):
        self.response_model = response_model

    def chat(self, *, inputs=None, agent_state=None, logger=None):
        type(self).last_inputs = inputs
        response = type("Response", (), {})()
        response.output_parsed = self.response_model(query="floating bed")
        return response, inputs, 0


def test_rag_benchmark_rewrites_query_then_searches(tmp_path, fake_wands_dataset):
    config_path = tmp_path / "rag.yml"
    config_path.write_text(
        """
strategy:
  name: rag_fixture
  type: rag
  params:
    model: gpt-5-mini
    system_prompt: Generate one concise search query.
    search_tools:
      - bm25:
""".lstrip(),
        encoding="utf-8",
    )

    with (
        patch("exps.runners.run.get_dataset", return_value=fake_wands_dataset),
        patch("exps.strategies.rag.build_openai_agent", _FakeQueryAgent),
    ):
        result = run_benchmark(
            RunParams(
                strategy_path=str(config_path),
                dataset="wands",
                num_queries=1,
                no_cache=True,
            )
        )

    assert result.strategy_name == "rag_fixture"
    assert result.summary["mean_ndcg"] > 0
    assert result.graded.iloc[0]["doc_id"] == 1
    assert _FakeQueryAgent.last_inputs[-1] == {
        "role": "user",
        "content": "floating bed",
    }
