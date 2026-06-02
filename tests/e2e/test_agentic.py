from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from exps.runners.run import RunParams, run_benchmark
from tests.utils.agent_fakes import FakeOpenAIAgent
from tests.utils.embedding_mocks import build_mock_embeddings


_TEMP_ROOT: Path | None = None


def _temp_root() -> Path:
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        _TEMP_ROOT = Path(tempfile.mkdtemp())
    return _TEMP_ROOT


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


def _fake_bash_builder(_corpus, **_kwargs):
    def bash(*_args, **_kwargs):
        return "ok"

    bash.__name__ = "bash"
    return bash


def _cleanup_temp_root() -> None:
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        return
    shutil.rmtree(_TEMP_ROOT, ignore_errors=True)
    _TEMP_ROOT = None


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


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
def test_agentic_hello_world_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    _paths_root,
    tmp_path,
    doug_blog_dataset,
):
    try:
        corpus = doug_blog_dataset.corpus
        judgments = doug_blog_dataset.judgments
        dataset_elapsed_s = 0.0
        embedding_elapsed_s = 0.0
        instances: list[FakeOpenAIAgent] = []

        def _mock_load_or_create_embeddings(corpus, passage_fn, **_kwargs):
            nonlocal embedding_elapsed_s
            embed_started_at = time.perf_counter()
            embeddings, model = build_mock_embeddings(
                corpus,
                judgments,
                passage_fn,
                dim=3,
                seed=123,
            )
            embedding_elapsed_s = time.perf_counter() - embed_started_at
            return embeddings, model

        mock_load_or_create_embeddings.side_effect = _mock_load_or_create_embeddings
        mock_load_model.side_effect = lambda *_args, **_kwargs: None

        doc_ids = [str(doc_id) for doc_id in corpus["doc_id"].head(3).tolist()]
        script = [
            {
                "function_call": {
                    "name": "search_embeddings",
                    "params": {"question": "salon chair", "top_k": 5},
                }
            },
            {"output": {"ranked_results": doc_ids}},
        ]

        config_path = tmp_path / "agentic_hello_world.yml"
        config_path.write_text(
            """
strategy:
  name: agentic_hello_world_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      You take user search queries and use a search tool to find products.
    search_tools:
      - e5_base_v2
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
        with patch(
            "exps.agentic.agent.build_openai_agent",
            side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
        ):
            started_at = time.perf_counter()
            result = run_benchmark(params)
            elapsed_s = time.perf_counter() - started_at

        assert result.metric_series is not None
        assert not result.metric_series.empty
        assert result.summary["tool_calls_mean"] >= 1.0
        assert sum(agent.chat_calls for agent in instances) >= 1
        assert mock_load_or_create_embeddings.call_count >= 1
        assert elapsed_s > 0.0

        benchmark = {
            "dataset_seconds": dataset_elapsed_s,
            "embedding_seconds": embedding_elapsed_s,
            "run_seconds": elapsed_s,
        }
        print(f"e2e_benchmark={benchmark}")

        trace_base = _paths_root / "agentic" / "doug_blog" / "agentic_hello_world_fixture"
        assert trace_base.exists()
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
def test_agentic_guarded_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    _paths_root,
    tmp_path,
    doug_blog_dataset,
):
    try:
        corpus = doug_blog_dataset.corpus
        judgments = doug_blog_dataset.judgments
        instances: list[FakeOpenAIAgent] = []
        doc_ids = [str(doc_id) for doc_id in corpus["doc_id"].head(3).tolist()]
        script = [
            {
                "function_call": {
                    "name": "search_bm25",
                    "params": {"keywords": "salon chair", "top_k": 5},
                }
            },
            {"output": {"ranked_results": doc_ids}},
        ]

        config_path = tmp_path / "agentic.yml"
        config_path.write_text(
            """
strategy:
  name: agentic_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      You take user search queries and use a search tool to find products.
    search_tools:
      - bm25:
          guards:
            - disallow_repeated_queries
      - embeddings:
          guards:
            - query_min_length:
                min_terms: 3
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
        with patch(
            "exps.agentic.agent.build_openai_agent",
            side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
        ):
            result = _run_with_embeddings(
                params,
                corpus=corpus,
                judgments=judgments,
                mock_load_or_create_embeddings=mock_load_or_create_embeddings,
                mock_load_model=mock_load_model,
            )

        assert result.metric_series is not None
        assert not result.metric_series.empty
        assert result.summary["tool_calls_mean"] >= 1.0
        assert sum(agent.chat_calls for agent in instances) >= 1
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.filesystem_index.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_agentic_filesystem_e2e(
    _filesystem_root,
    _paths_root,
    tmp_path,
    doug_blog_dataset,
):
    try:
        corpus = doug_blog_dataset.corpus
        instances: list[FakeOpenAIAgent] = []
        doc_ids = [str(doc_id) for doc_id in corpus["doc_id"].head(3).tolist()]
        script = [
            {
                "function_call": {
                    "name": "ls",
                    "params": {"path": ".", "glob": "**/*"},
                }
            },
            {"output": {"ranked_results": doc_ids}},
        ]

        config_path = tmp_path / "agentic_filesystem.yml"
        config_path.write_text(
            """
strategy:
  name: agentic_filesystem_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      You take user search queries and use filesystem tools to find the most relevant products.
      Use grep to find matching files, cat to read them, and then rank results.
    search_tools:
      - ls
      - grep
      - cat
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
        with patch(
            "exps.agentic.agent.build_openai_agent",
            side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
        ):
            result = run_benchmark(params)

        assert result.metric_series is not None
        assert not result.metric_series.empty
        assert sum(agent.chat_calls for agent in instances) >= 1

        trace_base = _paths_root / "agentic" / "doug_blog" / "agentic_filesystem_fixture"
        assert trace_base.exists()
    finally:
        _cleanup_temp_root()


@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
@patch.dict("os.environ", {"OPENAI_API_KEY": "stub"})
def test_agentic_query_rewrite_tool_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    tmp_path,
    doug_blog_dataset,
):
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    script = [{"output": {"ranked_results": doc_ids}}]

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
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
    ):
        result = _run_with_embeddings(
            params,
            corpus=doug_blog_dataset.corpus,
            judgments=doug_blog_dataset.judgments,
            mock_load_or_create_embeddings=mock_load_or_create_embeddings,
            mock_load_model=mock_load_model,
        )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert sum(agent.chat_calls for agent in instances) >= 1


@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
def test_agentic_orchestrate_bm25_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    tmp_path,
    doug_blog_dataset,
):
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    script = [{"output": {"ranked_results": doc_ids}}]

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
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
    ):
        result = _run_with_embeddings(
            params,
            corpus=doug_blog_dataset.corpus,
            judgments=doug_blog_dataset.judgments,
            mock_load_or_create_embeddings=mock_load_or_create_embeddings,
            mock_load_model=mock_load_model,
        )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert sum(agent.chat_calls for agent in instances) >= 1


@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
def test_agentic_plan_agents_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    tmp_path,
    doug_blog_dataset,
):
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    script = [{"output": {"ranked_results": doc_ids}}]

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
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
    ):
        result = _run_with_embeddings(
            params,
            corpus=doug_blog_dataset.corpus,
            judgments=doug_blog_dataset.judgments,
            mock_load_or_create_embeddings=mock_load_or_create_embeddings,
            mock_load_model=mock_load_model,
        )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert sum(agent.chat_calls for agent in instances) >= 1


@patch.dict(
    "exps.tools.registry.TOOL_REGISTRY",
    {"bash": {"builder": _fake_bash_builder, "kind": "agentic"}},
    clear=False,
)
def test_agentic_bash_tool_e2e(tmp_path, doug_blog_dataset):
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    script = [
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
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
    ):
        result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert sum(agent.chat_calls for agent in instances) >= 1


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


def test_agentic_few_shot_happy_path_e2e(tmp_path, doug_blog_dataset):
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    script = [
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
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
    ):
        result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert sum(agent.chat_calls for agent in instances) >= 1


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


def test_agentic_validator_tool_calls_e2e(tmp_path, doug_blog_dataset):
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    scripts = [
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
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(scripts=scripts, doc_ids=doc_ids, instances=instances),
    ):
        result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert len(instances) == 1
    assert instances[0].chat_calls == 3


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


def test_agentic_codegen_tool_e2e(tmp_path, doug_blog_dataset):
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    script = [
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
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
    ):
        result = run_benchmark(params)

    assert not result.metric_series.empty
    assert sum(agent.chat_calls for agent in instances) >= 1


@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
def test_agentic_codegen_fixture_nonzero_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    doug_blog_dataset,
):
    instances: list[FakeOpenAIAgent] = []
    doc_ids = [str(doc_id) for doc_id in doug_blog_dataset.corpus["doc_id"].head(3).tolist()]
    script = [
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
    with patch(
        "exps.agentic.agent.build_openai_agent",
        side_effect=_build_fake_agent(script=script, doc_ids=doc_ids, instances=instances),
    ):
        result = _run_with_embeddings(
            params,
            corpus=doug_blog_dataset.corpus,
            judgments=doug_blog_dataset.judgments,
            mock_load_or_create_embeddings=mock_load_or_create_embeddings,
            mock_load_model=mock_load_model,
        )

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert sum(agent.chat_calls for agent in instances) >= 1
