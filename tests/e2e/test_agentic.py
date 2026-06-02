from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from exps.datasets import get_dataset
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


def _set_fake_doc_ids(corpus, *, count: int = 3) -> None:
    FakeOpenAIAgent.calls = 0
    FakeOpenAIAgent.doc_ids = [str(doc_id) for doc_id in corpus["doc_id"].head(count).tolist()]


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
):
    try:
        dataset_started_at = time.perf_counter()
        dataset = get_dataset("doug_blog", ensure_snowball=False)
        corpus = dataset.corpus
        judgments = dataset.judgments
        dataset_elapsed_s = time.perf_counter() - dataset_started_at
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

        _set_fake_doc_ids(corpus)

        params = RunParams(
            strategy_path="configs/agentic_hello_world.yml",
            base_path="tests/fixtures",
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
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
@patch("exps.agentic.agent.OpenAIAgent", FakeOpenAIAgent)
def test_agentic_guarded_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    _paths_root,
):
    try:
        dataset = get_dataset("doug_blog", ensure_snowball=False)
        corpus = dataset.corpus
        judgments = dataset.judgments
        _set_fake_doc_ids(corpus)

        params = RunParams(
            strategy_path="configs/agentic.yml",
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
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.filesystem_index.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.agentic.agent.OpenAIAgent", FakeOpenAIAgent)
def test_agentic_filesystem_e2e(_filesystem_root, _paths_root):
    try:
        dataset = get_dataset("doug_blog", ensure_snowball=False)
        corpus = dataset.corpus
        _set_fake_doc_ids(corpus)

        params = RunParams(
            strategy_path="configs/agentic_filesystem.yml",
            base_path="tests/fixtures",
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
        _cleanup_temp_root()
