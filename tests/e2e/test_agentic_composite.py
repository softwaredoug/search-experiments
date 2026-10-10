from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from exps.runners.run import RunParams, run_benchmark
from tests.utils.agent_fakes import FakeOpenAIAgent


class _FakeEmbeddingModel:
    def encode(self, _text):
        return np.array([1.0, 0.0])


class _FakeJevClient:
    calls = []

    def __init__(self, *, api_key, model):
        self.api_key = api_key
        self.model = model
        type(self).calls = []

    def system_one(self, *, state, questions):
        type(self).calls.append((state, questions))
        return SimpleNamespace(
            answers={
                "relevance": SimpleNamespace(
                    probabilities={"Relevant": 0.9, "Not Relevant": 0.1},
                    confidence=0.95,
                )
            }
        )


def test_agentic_composite_rrf_e2e(tmp_path, doug_blog_dataset):
    corpus = doug_blog_dataset.corpus
    doc_ids = [str(doc_id) for doc_id in corpus["doc_id"].head(10).tolist()]
    instances: list[FakeOpenAIAgent] = []

    def build_fake_agent(*args, **kwargs):
        agent = FakeOpenAIAgent(*args, **kwargs)
        agent.scripts = [
            [
                {
                    "function_call": {
                        "name": "search_composite",
                        "params": {"query": "salon chair", "top_k": 5},
                    }
                },
                {"output": {"ranked_results": doc_ids}},
            ]
        ]
        agent.doc_ids = doc_ids
        instances.append(agent)
        return agent

    config_path = tmp_path / "agentic_composite_rrf.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_composite_rrf_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: Use the composite search tool to find relevant results.
    search_tools:
      - composite:
          retrieval_engine:
            type: rrf
            tools:
              - bm25
              - e5_base_v2
            weights: [2, 1]
            rank_constant: 60
          reranker_engine:
            type: jev
            k: 3
            params:
              decision_model: jev/jev-latest
              decision_weight: 10
              confidence_threshold: 0.7
              state_format: "{title}: {description}"
              prompt: "Is this document relevant to the {query}?"
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        batch_size=1,
        device="cpu",
        no_cache=True,
    )

    with (
        patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", tmp_path / "experiments"),
        patch(
            "exps.tools.embeddings.load_or_create_embeddings",
            return_value=(np.zeros((len(corpus), 2)), _FakeEmbeddingModel()),
        ) as mock_embeddings,
        patch("exps.tools.embeddings.load_model", return_value=_FakeEmbeddingModel()),
        patch("exps.agentic.agent.build_openai_agent", side_effect=build_fake_agent),
        patch("exps.tools.reranker.TypeSafeClient", _FakeJevClient),
        patch("exps.tools.reranker.key_for_provider", return_value="typesafe-test-key"),
    ):
        result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert result.summary["tool_calls_mean"] >= 1.0
    assert mock_embeddings.call_count == 1
    assert len(_FakeJevClient.calls) == 3
    assert len(instances) == 1
    tool_outputs = [
        item["output"]
        for item in instances[0].last_inputs
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert isinstance(tool_outputs[0], list)
    assert tool_outputs[0]
    assert {"id", "score"}.issubset(tool_outputs[0][0])
