from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from cheat_at_search.codegen.models import Edit
from exps.runners.train import TrainParams, train_strategy
from tests.utils.agent_fakes import FakeCodegenAgent
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


def _mock_embeddings(mock_load_or_create_embeddings, mock_load_model, *, dataset) -> None:
    judgments = dataset.judgments
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
    mock_load_model.side_effect = lambda *_args, **_kwargs: model_holder.get("model")


def _fielded_bm25_patch() -> Edit:
    return Edit(
        anchor="def reranker(",
        block_until="    return [str(doc['id']) for doc in docs]\n",
        action="replace",
        text=(
            "def reranker(query, top_k, fielded_bm25, **kwargs):\n"
            "    docs = fielded_bm25(query, fields=['title^99.0'], operator='or', top_k=top_k)\n"
            "    return [str(doc['id']) for doc in docs]\n"
        ),
        intention="Boost title weight",
        why="Title matches are more important than description matches.",
    )


def _fielded_bm25_minilm_patch() -> Edit:
    return Edit(
        anchor="def reranker(",
        block_until="    return [str(doc['id']) for doc in docs]\n",
        action="replace",
        text=(
            "def reranker(query, top_k, fielded_bm25, minilm, **kwargs):\n"
            "    docs = fielded_bm25(query, fields=['title^99.0'], operator='or', top_k=top_k)\n"
            "    return [str(doc['id']) for doc in docs]\n"
        ),
        intention="Boost title weight",
        why="Title matches are more important than description matches.",
    )


def _get_corpus_patch() -> Edit:
    return Edit(
        anchor="def reranker(",
        block_until="    return [str(doc['id']) for doc in docs]\n",
        action="replace",
        text=(
            "def reranker(query, top_k, get_corpus, **kwargs):\n"
            "    corpus = get_corpus()\n"
            "    docs = corpus.head(top_k).to_dict('records')\n"
            "    return [str(doc['doc_id']) for doc in docs]\n"
        ),
        intention="Use corpus head",
        why="Validate raw tool compatibility.",
    )


def _get_corpus_with_bm25_patch() -> Edit:
    return Edit(
        anchor="def reranker(",
        block_until="    return [str(doc['id']) for doc in docs]\n",
        action="replace",
        text=(
            "def reranker(query, top_k, bm25, get_corpus, **kwargs):\n"
            "    corpus = get_corpus()\n"
            "    docs = corpus.head(top_k).to_dict('records')\n"
            "    return [str(doc['doc_id']) for doc in docs]\n"
        ),
        intention="Use corpus head",
        why="Validate raw tool compatibility.",
    )


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.codegen.train.OpenAIAgent", FakeCodegenAgent)
def test_codegen_commit_known_good_patch_e2e(_paths_root, tmp_path):
    try:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        config_path = tmp_path / "codegen.yml"
        config_path.write_text(
            f"""
strategy:
  name: codegen_known_good_fixture
  type: codegen
  path: {run_dir}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - fielded_bm25
      eval:
        train_fraction: 0.2
        seed: 123
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.
    run:
      top_k: 5
""".lstrip(),
            encoding="utf-8",
        )

        patch_edit = _fielded_bm25_patch()
        original_patch = FakeCodegenAgent.patch_edit
        FakeCodegenAgent.patch_edit = patch_edit
        try:
            params = TrainParams(
                strategy_path=str(config_path),
                base_path=None,
                dataset="doug_blog",
                num_queries=1,
                seed=123,
                workers=1,
                device=None,
                rounds=1,
            )
            result = train_strategy(params)
        finally:
            FakeCodegenAgent.patch_edit = original_patch

        assert result.artifact_path
        code_path = Path(result.artifact_path) / "reranker.py"
        assert code_path.exists()
        code = code_path.read_text(encoding="utf-8")
        assert "fields=['title^99.0']" in code
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
@patch("exps.codegen.train.OpenAIAgent", FakeCodegenAgent)
def test_codegen_guarded_train_e2e(
    mock_load_model,
    mock_load_or_create_embeddings,
    _paths_root,
    tmp_path,
    doug_blog_dataset,
):
    try:
        _mock_embeddings(
            mock_load_or_create_embeddings,
            mock_load_model,
            dataset=doug_blog_dataset,
        )
        config_path = tmp_path / "codegen_guarded_train.yml"
        config_path.write_text(
            """
strategy:
  name: codegen_guarded_train_fixture
  type: codegen
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - fielded_bm25
        - minilm
      edit:
        guards:
          - validation
          - length:
              max_lines: 5
              max_cols: 120
      eval:
        train_fraction: 0.2
        seed: 123
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.

    run:
      top_k: 5
""".lstrip(),
            encoding="utf-8",
        )
        patch_edit = _fielded_bm25_minilm_patch()
        original_patch = FakeCodegenAgent.patch_edit
        FakeCodegenAgent.patch_edit = patch_edit
        try:
            params = TrainParams(
                strategy_path=str(config_path),
                base_path=None,
                dataset="doug_blog",
                num_queries=1,
                seed=123,
                workers=1,
                device=None,
                rounds=1,
            )
            result = train_strategy(params)
        finally:
            FakeCodegenAgent.patch_edit = original_patch

        assert result.artifact_path
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.codegen.train.OpenAIAgent", FakeCodegenAgent)
def test_codegen_get_corpus_e2e(_paths_root, tmp_path):
    try:
        patch_edit = _get_corpus_with_bm25_patch()
        original_patch = FakeCodegenAgent.patch_edit
        FakeCodegenAgent.patch_edit = patch_edit
        try:
            params = TrainParams(
                strategy_path="configs/codegen_get_corpus.yml",
                base_path="tests/fixtures",
                dataset="doug_blog",
                num_queries=1,
                seed=123,
                workers=1,
                device=None,
                rounds=1,
            )
            result = train_strategy(params)
        finally:
            FakeCodegenAgent.patch_edit = original_patch

        assert result.artifact_path
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.codegen.train.OpenAIAgent", FakeCodegenAgent)
def test_codegen_raw_only_e2e(_paths_root, tmp_path):
    try:
        patch_edit = _get_corpus_patch()
        original_patch = FakeCodegenAgent.patch_edit
        FakeCodegenAgent.patch_edit = patch_edit
        try:
            params = TrainParams(
                strategy_path="configs/codegen_raw_only.yml",
                base_path="tests/fixtures",
                dataset="doug_blog",
                num_queries=1,
                seed=123,
                workers=1,
                device=None,
                rounds=1,
            )
            result = train_strategy(params)
        finally:
            FakeCodegenAgent.patch_edit = original_patch

        assert result.artifact_path
    finally:
        _cleanup_temp_root()
