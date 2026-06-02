from __future__ import annotations

from unittest.mock import patch

from exps.runners.run import RunParams, run_benchmark
from tests.utils.agent_fakes import FakeOpenAIAgent
from tests.utils.embedding_mocks import build_mock_embeddings


def _build_fake_agent(
    *,
    script: list[dict] | None = None,
    scripts: list[list[dict]] | None = None,
    doc_ids: list[str] | None = None,
    categories: list[str] | None = None,
    instances: list[FakeOpenAIAgent] | None = None,
):
    resolved_scripts = scripts
    if resolved_scripts is None and script is not None:
        resolved_scripts = [script]

    def _factory(*args, **kwargs):
        agent = FakeOpenAIAgent(*args, **kwargs)
        agent.scripts = resolved_scripts
        if doc_ids is not None:
            agent.doc_ids = list(doc_ids)
        if categories is not None:
            agent.categories = list(categories)
        if instances is not None:
            instances.append(agent)
        return agent

    return _factory


def _run_with_wands_embeddings(params, *, corpus, judgments, mock_load_or_create_embeddings, mock_load_model):
    model_holder: dict[str, object] = {}

    def _mock_load_or_create_embeddings(corpus, passage_fn, **_kwargs):
        embeddings, model = build_mock_embeddings(
            corpus,
            judgments,
            passage_fn,
            dim=3,
            seed=123,
        )
        model_holder["model"] = model
        return embeddings, model

    mock_load_or_create_embeddings.side_effect = _mock_load_or_create_embeddings
    mock_load_model.side_effect = lambda *_args, **_kwargs: model_holder["model"]
    return run_benchmark(params)


@patch("exps.runners.run.get_dataset")
@patch("exps.tools.wands.load_or_create_embeddings")
@patch("exps.tools.wands.load_model")
def test_agentic_wands_bm25_e5_few_shot_delegate_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    mock_get_dataset,
    fake_wands_dataset,
):
    mock_get_dataset.return_value = fake_wands_dataset
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in fake_wands_dataset.corpus["doc_id"].head(3).tolist()]
    script = [
        {
            "function_call": {
                "name": "search_bm25_wands",
                "params": {"keywords": "floating bed", "top_k": 5},
            }
        },
        {"output": {"ranked_results": doc_ids}},
    ]

    params = RunParams(
        strategy_path="configs/agentic_wands_bm25_e5_few_shot_delegate.yml",
        base_path="tests/fixtures",
        dataset="wands",
        num_queries=1,
        seed=123,
        workers=1,
        batch_size=1,
        device=None,
        no_cache=True,
    )
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
    ):
        result = _run_with_wands_embeddings(
            params,
            corpus=fake_wands_dataset.corpus,
            judgments=fake_wands_dataset.judgments,
            mock_load_or_create_embeddings=mock_load_or_create_embeddings,
            mock_load_model=mock_load_model,
        )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert sum(agent.chat_calls for agent in instances) >= 1


@patch("exps.runners.run.get_dataset")
@patch("exps.tools.wands.load_or_create_embeddings")
@patch("exps.tools.wands.load_model")
def test_scatter_gather_wands_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    mock_get_dataset,
    fake_wands_dataset,
):
    mock_get_dataset.return_value = fake_wands_dataset
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in fake_wands_dataset.corpus["doc_id"].head(3).tolist()]
    categories = fake_wands_dataset.corpus["cat_subcat"].dropna().astype(str).unique().tolist()
    script = [{"output": {"categories": categories[:2], "ranked_results": doc_ids}}]

    params = RunParams(
        strategy_path="configs/scatter_gather_wands.yml",
        base_path="tests/fixtures",
        dataset="wands",
        num_queries=1,
        seed=123,
        workers=1,
        batch_size=1,
        device=None,
        no_cache=True,
    )
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(
            script=script,
            doc_ids=doc_ids,
            categories=categories[:2],
            instances=instances,
        ),
    ):
        result = _run_with_wands_embeddings(
            params,
            corpus=fake_wands_dataset.corpus,
            judgments=fake_wands_dataset.judgments,
            mock_load_or_create_embeddings=mock_load_or_create_embeddings,
            mock_load_model=mock_load_model,
        )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert sum(agent.chat_calls for agent in instances) >= 1


@patch("exps.runners.run.get_dataset")
@patch("exps.tools.wands.load_or_create_embeddings")
@patch("exps.tools.wands.load_model")
def test_scatter_gather_wands_cat_subcat_query_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    mock_get_dataset,
    fake_wands_dataset,
):
    mock_get_dataset.return_value = fake_wands_dataset
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in fake_wands_dataset.corpus["doc_id"].head(3).tolist()]
    categories = fake_wands_dataset.corpus["cat_subcat"].dropna().astype(str).unique().tolist()
    script = [{"output": {"categories": categories[:2], "ranked_results": doc_ids}}]

    params = RunParams(
        strategy_path="configs/scatter_gather_wands_cat_subcat.yml",
        base_path="tests/fixtures",
        dataset="wands",
        query="floating bed",
        k=5,
        seed=123,
        workers=1,
        batch_size=1,
        device=None,
        no_cache=True,
    )
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(
            script=script,
            doc_ids=doc_ids,
            categories=categories[:2],
            instances=instances,
        ),
    ):
        result = _run_with_wands_embeddings(
            params,
            corpus=fake_wands_dataset.corpus,
            judgments=fake_wands_dataset.judgments,
            mock_load_or_create_embeddings=mock_load_or_create_embeddings,
            mock_load_model=mock_load_model,
        )

    assert result.query_results is not None
    assert not result.query_results.empty
    assert sum(agent.chat_calls for agent in instances) >= 1
