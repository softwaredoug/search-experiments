from __future__ import annotations

from unittest.mock import patch

import pytest
from exps.runners.run import RunParams, run_benchmark
from tests.utils.agent_fakes import FakeOpenAIAgent
from tests.utils.embedding_mocks import build_mock_embeddings


def _fake_bash_builder(_corpus, **_kwargs):
    def bash(*_args, **_kwargs):
        return "ok"

    bash.__name__ = "bash"
    return bash


def _build_fake_agent(*args, **kwargs):
    return FakeOpenAIAgent(*args, **kwargs)


def _run_with_embeddings(params, *, corpus, judgments, mock_load_or_create_embeddings, mock_load_model):
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
@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
def test_agentic_wands_bm25_e5_few_shot_delegate_e2e(
    _build_agent,
    mock_load_model,
    mock_load_or_create_embeddings,
    mock_get_dataset,
    fake_wands_dataset,
):
    mock_get_dataset.return_value = fake_wands_dataset
    FakeOpenAIAgent.calls = 0
    doc_ids = [str(doc_id) for doc_id in fake_wands_dataset.corpus["doc_id"].head(3).tolist()]
    original_script = FakeOpenAIAgent.script
    FakeOpenAIAgent.script = [
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
    result = _run_with_wands_embeddings(
        params,
        corpus=fake_wands_dataset.corpus,
        judgments=fake_wands_dataset.judgments,
        mock_load_or_create_embeddings=mock_load_or_create_embeddings,
        mock_load_model=mock_load_model,
    )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert FakeOpenAIAgent.calls >= 1
    FakeOpenAIAgent.script = original_script


@patch("exps.runners.run.get_dataset")
@patch("exps.tools.wands.load_or_create_embeddings")
@patch("exps.tools.wands.load_model")
@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
def test_scatter_gather_wands_e2e(
    _build_agent,
    mock_load_model,
    mock_load_or_create_embeddings,
    mock_get_dataset,
    fake_wands_dataset,
):
    mock_get_dataset.return_value = fake_wands_dataset
    FakeOpenAIAgent.calls = 0
    doc_ids = [str(doc_id) for doc_id in fake_wands_dataset.corpus["doc_id"].head(3).tolist()]
    categories = fake_wands_dataset.corpus["cat_subcat"].dropna().astype(str).unique().tolist()
    original_script = FakeOpenAIAgent.script
    FakeOpenAIAgent.script = [
        {"output": {"categories": categories[:2], "ranked_results": doc_ids}}
    ]

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
    result = _run_with_wands_embeddings(
        params,
        corpus=fake_wands_dataset.corpus,
        judgments=fake_wands_dataset.judgments,
        mock_load_or_create_embeddings=mock_load_or_create_embeddings,
        mock_load_model=mock_load_model,
    )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert FakeOpenAIAgent.calls >= 1
    FakeOpenAIAgent.script = original_script


@patch("exps.runners.run.get_dataset")
@patch("exps.tools.wands.load_or_create_embeddings")
@patch("exps.tools.wands.load_model")
@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
def test_scatter_gather_wands_cat_subcat_query_e2e(
    _build_agent,
    mock_load_model,
    mock_load_or_create_embeddings,
    mock_get_dataset,
    fake_wands_dataset,
):
    mock_get_dataset.return_value = fake_wands_dataset
    FakeOpenAIAgent.calls = 0
    doc_ids = [str(doc_id) for doc_id in fake_wands_dataset.corpus["doc_id"].head(3).tolist()]
    categories = fake_wands_dataset.corpus["cat_subcat"].dropna().astype(str).unique().tolist()
    original_script = FakeOpenAIAgent.script
    FakeOpenAIAgent.script = [
        {"output": {"categories": categories[:2], "ranked_results": doc_ids}}
    ]

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
    result = _run_with_wands_embeddings(
        params,
        corpus=fake_wands_dataset.corpus,
        judgments=fake_wands_dataset.judgments,
        mock_load_or_create_embeddings=mock_load_or_create_embeddings,
        mock_load_model=mock_load_model,
    )

    assert result.query_results is not None
    assert not result.query_results.empty
    assert FakeOpenAIAgent.calls >= 1
    FakeOpenAIAgent.script = original_script


@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
@patch.dict("os.environ", {"OPENAI_API_KEY": "stub"})
def test_agentic_query_rewrite_tool_e2e(
    _build_agent,
    mock_load_model,
    mock_load_or_create_embeddings,
    tmp_path,
    doug_blog_dataset,
):
    FakeOpenAIAgent.calls = 0
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    original_script = FakeOpenAIAgent.script
    FakeOpenAIAgent.script = [{"output": {"ranked_results": doc_ids}}]

    config_path = tmp_path / "agentic_query_rewrite_e2e.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_query_rewrite_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      You take user search queries and use search tools to find the most relevant products.
    search_tools:
      - query_rewrite:
          model: gpt-5-mini
          max_alternatives: 2
      - bm25
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
        device=None,
        no_cache=True,
    )
    result = _run_with_embeddings(
        params,
        corpus=doug_blog_dataset.corpus,
        judgments=doug_blog_dataset.judgments,
        mock_load_or_create_embeddings=mock_load_or_create_embeddings,
        mock_load_model=mock_load_model,
    )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    FakeOpenAIAgent.script = original_script


@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
def test_agentic_orchestrate_bm25_e2e(
    _build_agent,
    mock_load_model,
    mock_load_or_create_embeddings,
    tmp_path,
    doug_blog_dataset,
):
    FakeOpenAIAgent.calls = 0
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    original_script = FakeOpenAIAgent.script
    FakeOpenAIAgent.script = [{"output": {"ranked_results": doc_ids}}]

    config_path = tmp_path / "agentic_orchestrate.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_orchestrate_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      You take user search queries and orchestrate subagents to find relevant products.
    subagent_system_prompt: |
      You help with tasks searchinging / finding content as instructed.
    search_tools:
      - delegate_task
      - bm25
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        batch_size=1,
        device=None,
        no_cache=True,
    )
    result = _run_with_embeddings(
        params,
        corpus=doug_blog_dataset.corpus,
        judgments=doug_blog_dataset.judgments,
        mock_load_or_create_embeddings=mock_load_or_create_embeddings,
        mock_load_model=mock_load_model,
    )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert FakeOpenAIAgent.calls >= 1
    FakeOpenAIAgent.script = original_script


@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
def test_agentic_plan_agents_e2e(
    _build_agent,
    mock_load_model,
    mock_load_or_create_embeddings,
    tmp_path,
    doug_blog_dataset,
):
    FakeOpenAIAgent.calls = 0
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    original_script = FakeOpenAIAgent.script
    FakeOpenAIAgent.script = [{"output": {"ranked_results": doc_ids}}]

    config_path = tmp_path / "agentic_plan.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_plan_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    agents:
      planning:
        system_prompt: |
          You plan how to search for relevant products.
        search_tools:
          - delegate_task
          - bm25
      search:
        system_prompt: |
          You find relevant products and return ranked DOC IDs.
        search_tools:
          - bm25
    plan:
      - planning: plan how to best search for {query}
      - search: find the most relevant results for {query}
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        batch_size=1,
        device=None,
        no_cache=True,
    )
    result = _run_with_embeddings(
        params,
        corpus=doug_blog_dataset.corpus,
        judgments=doug_blog_dataset.judgments,
        mock_load_or_create_embeddings=mock_load_or_create_embeddings,
        mock_load_model=mock_load_model,
    )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    FakeOpenAIAgent.script = original_script


@patch.dict(
    "exps.tools.registry.TOOL_REGISTRY",
    {"bash": {"builder": _fake_bash_builder, "kind": "agentic"}},
    clear=False,
)
@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
def test_agentic_bash_tool_e2e(_build_agent, tmp_path, doug_blog_dataset):
    FakeOpenAIAgent.calls = 0
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    original_script = FakeOpenAIAgent.script
    FakeOpenAIAgent.script = [
        {
            "function_call": {
                "name": "bash",
                "params": None,
            }
        },
        {"output": {"ranked_results": doc_ids}},
    ]

    config_path = tmp_path / "agentic_bash.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_bash_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use the bash tool to search /corpus for relevant products.
    search_tools:
      - bash
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
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty
    FakeOpenAIAgent.script = original_script


def test_agentic_raw_tool_rejected_e2e(tmp_path):
    config_path = tmp_path / "agentic_raw_tool.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_raw_tool_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use the search tool to find products.
    search_tools:
      - get_corpus
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
        device=None,
        no_cache=True,
    )
    with pytest.raises(ValueError, match="raw search tool"):
        run_benchmark(params)


def test_agentic_dataset_specific_tool_rejected_e2e(tmp_path):
    config_path = tmp_path / "agentic_wands_tool.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_wands_tool_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use the search tool to find products.
    search_tools:
      - bm25_wands
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
        device=None,
        no_cache=True,
    )
    with pytest.raises(ValueError, match="only available for wands dataset"):
        run_benchmark(params)


@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
def test_agentic_few_shot_happy_path_e2e(_build_agent, tmp_path, doug_blog_dataset):
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    original_script = FakeOpenAIAgent.script
    FakeOpenAIAgent.script = [
        {
            "function_call": {
                "name": "search_bm25",
                "params": {"keywords": "salon chair", "top_k": 5},
            }
        },
        {"output": {"ranked_results": doc_ids}},
    ]
    config_path = tmp_path / "agentic_few_shot.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_few_shot_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use search tools to find products.
    few_shot:
      - sample_judgments:
          num_rows: 4
    search_tools:
      - bm25
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
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty
    FakeOpenAIAgent.script = original_script


def test_agentic_few_shot_missing_column_raises_e2e(tmp_path):
    config_path = tmp_path / "agentic_few_shot_bad_col.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_few_shot_bad_col_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use search tools to find products.
    few_shot:
      - sample_judgments:
          num_rows: 4
          columns:
            - missing_col
    search_tools:
      - bm25
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
        device=None,
        no_cache=True,
    )
    with pytest.raises(ValueError, match="few_shot column not found"):
        run_benchmark(params)


@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
def test_agentic_validator_tool_calls_e2e(_build_agent, tmp_path, doug_blog_dataset):
    FakeOpenAIAgent.calls = 0
    FakeOpenAIAgent.chat_calls = 0
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    original_script = FakeOpenAIAgent.script
    original_scripts = FakeOpenAIAgent.scripts
    FakeOpenAIAgent.scripts = [
        [
            {
                "function_call": {
                    "name": "search_bm25",
                    "params": {"keywords": "salon chair", "top_k": 5},
                }
            },
            {"output": {"ranked_results": doc_ids}},
        ],
        [
            {
                "function_call": {
                    "name": "search_bm25",
                    "params": {"keywords": "salon chair", "top_k": 5},
                }
            },
            {"output": {"ranked_results": doc_ids}},
        ],
        [
            {
                "function_call": {
                    "name": "search_bm25",
                    "params": {"keywords": "salon chair", "top_k": 5},
                }
            },
            {"output": {"ranked_results": doc_ids}},
        ],
    ]
    try:
        config_path = tmp_path / "agentic_validator_tool_calls.yml"
        config_path.write_text(
            """
strategy:
  name: agentic_validator_tool_calls_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use search tools to find products.
    search_tools:
      - bm25
    validators:
      - tool_calls:
          prompt: "Keep using tools until you hit 3 calls."
          params:
            num_calls: 3
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
            device=None,
            no_cache=True,
        )
        result = run_benchmark(params)

        assert result.metric_series is not None
        assert not result.metric_series.empty
        assert FakeOpenAIAgent.chat_calls == 3
        assert FakeOpenAIAgent.last_instance is not None
        assert FakeOpenAIAgent.last_instance.chat_calls == 3
    finally:
        FakeOpenAIAgent.script = original_script
        FakeOpenAIAgent.scripts = original_scripts


def test_agentic_codegen_tool_dependency_mismatch_e2e(tmp_path):
    reranker_dir = tmp_path / "codegen_dependency_mismatch"
    reranker_dir.mkdir()
    reranker_path = reranker_dir / "reranker.py"
    reranker_path.write_text(
        """
def rerank_doug_blog(query, fielded_bm25, **kwargs):
    docs = fielded_bm25(
        query,
        fields=['title^9.3', 'description^4.1'],
        operator='or',
        top_k=5,
    )
    return [doc['id'] for doc in docs]
""".lstrip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "agentic_codegen_dep.yml"
    config_path.write_text(
        f"""
strategy:
  name: agentic_codegen_dep_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use search tools to find products.
    search_tools:
      - codegen:
          path: {reranker_dir}
          name: search
          dependencies:
            - bm25
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
        device=None,
        no_cache=True,
    )
    with pytest.raises(ValueError, match="codegen tool missing dependencies"):
        run_benchmark(params)


def test_agentic_codegen_tool_return_fields_validation_e2e(tmp_path):
    reranker_dir = tmp_path / "codegen_return_fields"
    reranker_dir.mkdir()
    reranker_path = reranker_dir / "reranker.py"
    reranker_path.write_text(
        """
def rerank_doug_blog(query, bm25, **kwargs):
    docs = bm25(query, top_k=5)
    return [doc['id'] for doc in docs]
""".lstrip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "agentic_codegen_return_fields.yml"
    config_path.write_text(
        f"""
strategy:
  name: agentic_codegen_return_fields_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use search tools to find products.
    search_tools:
      - codegen:
          path: {reranker_dir}
          name: search
          dependencies:
            - bm25
          return_fields:
            - missing_col
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
        device=None,
        no_cache=True,
    )
    with pytest.raises(ValueError, match="return_fields not found in corpus"):
        run_benchmark(params)


@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
def test_agentic_codegen_tool_e2e(_build_agent, tmp_path, doug_blog_dataset):
    FakeOpenAIAgent.calls = 0
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    original_script = FakeOpenAIAgent.script
    FakeOpenAIAgent.script = [
        {
            "function_call": {
                "name": "search",
                "params": {"query": "salon chair", "top_k": 10},
            }
        },
        {"output": {"ranked_results": doc_ids}},
    ]
    codegen_dir = tmp_path / "codegen_run"
    codegen_dir.mkdir()
    reranker_path = codegen_dir / "reranker.py"
    reranker_path.write_text(
        """
def rerank_wands(query, fielded_bm25, **kwargs):
    docs = fielded_bm25(
        keywords=query,
        fields=['title^9.3', 'description^4.1'],
        operator='or',
        top_k=10,
    )
    return [doc['id'] for doc in docs]
""".lstrip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "agentic_codegen.yml"
    config_path.write_text(
        f"""
strategy:
  name: agentic_codegen_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      You take user search queries and use search tools to find the most relevant products.
    search_tools:
      - codegen:
          path: {codegen_dir}
          name: search
          dependencies:
            - fielded_bm25
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
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert not result.metric_series.empty
    FakeOpenAIAgent.script = original_script


@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
@patch("exps.agentic.agent.build_openai_agent", side_effect=_build_fake_agent)
def test_agentic_codegen_fixture_nonzero_e2e(
    _build_agent,
    mock_load_model,
    mock_load_or_create_embeddings,
    doug_blog_dataset,
):
    FakeOpenAIAgent.calls = 0
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    original_script = FakeOpenAIAgent.script
    FakeOpenAIAgent.script = [
        {
            "function_call": {
                "name": "search",
                "params": {"query": "salon chair", "top_k": 5},
            }
        },
        {"output": {"ranked_results": doc_ids}},
    ]

    params = RunParams(
        strategy_path="configs/agentic_w_codegen.yml",
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=2,
        seed=123,
        workers=1,
        batch_size=1,
        device=None,
        no_cache=True,
    )
    result = _run_with_embeddings(
        params,
        corpus=doug_blog_dataset.corpus,
        judgments=doug_blog_dataset.judgments,
        mock_load_or_create_embeddings=mock_load_or_create_embeddings,
        mock_load_model=mock_load_model,
    )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    FakeOpenAIAgent.script = original_script
