from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from exps.runners.run import RunParams, run_benchmark
from tests.utils.agent_fakes import FakeOpenAIAgent
from tests.utils.embedding_mocks import build_mock_embeddings


_TEMP_ROOT: Path | None = None


def _temp_root() -> Path:
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        _TEMP_ROOT = Path(tempfile.mkdtemp())
    return _TEMP_ROOT


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
@patch("exps.agentic.agent.OpenAIAgent", FakeOpenAIAgent)
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

        FakeOpenAIAgent.calls = 0
        doc_ids = [str(doc_id) for doc_id in corpus["doc_id"].head(3).tolist()]
        original_script = FakeOpenAIAgent.script
        FakeOpenAIAgent.script = [
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
        started_at = time.perf_counter()
        result = run_benchmark(params)
        elapsed_s = time.perf_counter() - started_at

        assert result.metric_series is not None
        assert not result.metric_series.empty
        assert result.summary["tool_calls_mean"] >= 1.0
        assert FakeOpenAIAgent.calls >= 1
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
        FakeOpenAIAgent.script = original_script
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
@patch("exps.agentic.agent.OpenAIAgent", FakeOpenAIAgent)
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
        FakeOpenAIAgent.calls = 0
        doc_ids = [str(doc_id) for doc_id in corpus["doc_id"].head(3).tolist()]
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
        assert FakeOpenAIAgent.calls >= 1
    finally:
        FakeOpenAIAgent.script = original_script
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.filesystem_index.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.agentic.agent.OpenAIAgent", FakeOpenAIAgent)
def test_agentic_filesystem_e2e(_filesystem_root, _paths_root, tmp_path, doug_blog_dataset):
    try:
        corpus = doug_blog_dataset.corpus
        FakeOpenAIAgent.calls = 0
        doc_ids = [str(doc_id) for doc_id in corpus["doc_id"].head(3).tolist()]
        original_script = FakeOpenAIAgent.script
        FakeOpenAIAgent.script = [
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
        result = run_benchmark(params)

        assert result.metric_series is not None
        assert not result.metric_series.empty
        assert FakeOpenAIAgent.calls >= 1

        trace_base = _paths_root / "agentic" / "doug_blog" / "agentic_filesystem_fixture"
        assert trace_base.exists()
    finally:
        FakeOpenAIAgent.script = original_script
        _cleanup_temp_root()
