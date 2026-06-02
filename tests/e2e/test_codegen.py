from __future__ import annotations

import json
import shutil
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from cheat_at_search.codegen.models import Edit
from exps.runners.run import RunParams, run_benchmark
from exps.runners.train import TrainParams, train_strategy
from tests.utils.agent_fakes import FakeOpenAIAgent
from tests.utils.embedding_mocks import build_mock_embeddings


_TEMP_ROOT: Path | None = None

_CODEGEN_FIXTURES = Path("tests/fixtures/codegen")


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


def _fixture_code(path: Path, *, indent: str = "        ") -> str:
    code = (_CODEGEN_FIXTURES / path).read_text(encoding="utf-8").rstrip()
    return textwrap.indent(code, indent)


def _load_rounds(path: Path) -> list[dict]:
    rounds_path = path / "rounds.jsonl"
    payload = rounds_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in payload if line.strip()]


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


def _description_only_patch() -> Edit:
    return Edit(
        anchor="def reranker(",
        block_until="    return [str(doc['id']) for doc in docs]\n",
        action="replace",
        text=(
            "def reranker(query, top_k, fielded_bm25, **kwargs):\n"
            "    docs = fielded_bm25(query, fields=['description^99.0'], operator='or', top_k=top_k)\n"
            "    return [str(doc['id']) for doc in docs]\n"
        ),
        intention="Only search descriptions",
        why="Deprioritize title phrase matches.",
    )


def _bm25_patch() -> Edit:
    return Edit(
        anchor="def reranker(",
        block_until="    return [str(doc['id']) for doc in docs]\n",
        action="replace",
        text=(
            "def reranker(query, top_k, bm25, **kwargs):\n"
            "    docs = bm25(query, top_k=top_k)\n"
            "    return [str(doc['id']) for doc in docs]\n"
        ),
        intention="Use bm25",
        why="Keep baseline behavior with bm25.",
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


def _commit_script(edit: Edit, *, message: str = "Done") -> list[dict]:
    return [
        {"function_call": {"name": "commit_patch", "params": edit}},
        {"output": {"message": message, "short_name": "patch", "summary": "Applied patch"}},
    ]


def _build_codegen_agent(
    *,
    script: list[dict] | None = None,
    scripts: list[list[dict]] | None = None,
    instances: list[FakeOpenAIAgent] | None = None,
):
    resolved_scripts = scripts
    if resolved_scripts is None and script is not None:
        resolved_scripts = [script]

    def _factory(*args, **kwargs):
        agent = FakeOpenAIAgent(*args, **kwargs)
        agent.scripts = resolved_scripts
        if instances is not None:
            instances.append(agent)
        return agent

    return _factory


def _with_codegen_agent(*, script: list[dict] | None = None, scripts: list[list[dict]] | None = None):
    return patch("exps.codegen.train.OpenAIAgent", side_effect=_build_codegen_agent(script=script, scripts=scripts))


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
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

        script = _commit_script(_fielded_bm25_patch())
        with _with_codegen_agent(script=script):
            params = TrainParams(
                strategy_path=str(config_path),
                base_path=None,
                dataset="doug_blog",
                num_queries=3,
                seed=123,
                workers=1,
                device=None,
                rounds=1,
            )
            result = train_strategy(params)

        assert result.artifact_path
        code_path = Path(result.artifact_path) / "reranker.py"
        assert code_path.exists()
        code = code_path.read_text(encoding="utf-8")
        assert "fields=['title^99.0']" in code
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_bad_ranker_rejected_e2e(_paths_root, tmp_path):
    try:
        run_dir = tmp_path / "codegen_bad_ranker"
        run_dir.mkdir()
        config_path = tmp_path / "codegen_bad_ranker.yml"
        config_path.write_text(
            f"""
strategy:
  name: codegen_bad_ranker_fixture
  type: codegen
  path: {run_dir}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      rounds: 1
      refresh_every: 1
      search_tools:
        - fielded_bm25
      start_code: |
{_fixture_code(Path("doug_blog_good_reranker.py"))}
      edit:
        guards:
          - validation
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

        script = _commit_script(_description_only_patch())
        with _with_codegen_agent(script=script):
            params = TrainParams(
                strategy_path=str(config_path),
                base_path=None,
                dataset="doug_blog",
                num_queries=3,
                seed=123,
                workers=1,
                device=None,
                rounds=1,
            )
            result = train_strategy(params)

        assert result.artifact_path
        code_path = Path(result.artifact_path) / "reranker.py"
        code = code_path.read_text(encoding="utf-8")
        assert "description^99.0" not in code
        assert "title^99.0" in code
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_good_ranker_applied_e2e(_paths_root, tmp_path):
    try:
        run_dir = tmp_path / "codegen_good_ranker"
        run_dir.mkdir()
        config_path = tmp_path / "codegen_good_ranker.yml"
        config_path.write_text(
            f"""
strategy:
  name: codegen_good_ranker_fixture
  type: codegen
  path: {run_dir}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      rounds: 1
      refresh_every: 1
      search_tools:
        - fielded_bm25
      start_code: |
{_fixture_code(Path("doug_blog_bad_reranker.py"))}
      edit:
        guards:
          - validation
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

        script = _commit_script(_fielded_bm25_patch())
        with _with_codegen_agent(script=script):
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

        assert result.artifact_path
        code_path = Path(result.artifact_path) / "reranker.py"
        code = code_path.read_text(encoding="utf-8")
        assert "fields=['title^99.0']" in code
        assert "description^4.1" not in code
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.embeddings.load_or_create_embeddings")
@patch("exps.tools.embeddings.load_model")
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
        script = _commit_script(_fielded_bm25_minilm_patch())
        with _with_codegen_agent(script=script):
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

        assert result.artifact_path
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_get_corpus_e2e(_paths_root, tmp_path):
    try:
        script = _commit_script(_get_corpus_with_bm25_patch())
        with _with_codegen_agent(script=script):
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

        assert result.artifact_path
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_raw_only_e2e(_paths_root, tmp_path):
    try:
        script = _commit_script(_get_corpus_patch())
        with _with_codegen_agent(script=script):
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

        assert result.artifact_path
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_start_code_e2e(_paths_root, tmp_path):
    try:
        script = _commit_script(_bm25_patch())
        with _with_codegen_agent(script=script):
            params = TrainParams(
                strategy_path="configs/codegen_start_code.yml",
                base_path="tests/fixtures",
                dataset="doug_blog",
                num_queries=1,
                seed=123,
                workers=1,
                device=None,
                rounds=1,
            )
            result = train_strategy(params)

        assert result.artifact_path
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_start_code_mismatch_e2e(_paths_root, tmp_path):
    try:
        params = TrainParams(
            strategy_path="configs/codegen_start_code_mismatch.yml",
            base_path="tests/fixtures",
            dataset="doug_blog",
            num_queries=1,
            seed=123,
            workers=1,
            device=None,
            rounds=1,
        )
        with pytest.raises(ValueError, match="start_code does not match configured tools"):
            train_strategy(params)
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_start_code_dedent_e2e(_paths_root, tmp_path):
    try:
        config_path = tmp_path / "codegen_start_code_dedent.yml"
        config_path.write_text(
            """
strategy:
  name: codegen_start_code_dedent_fixture
  type: codegen
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - bm25
      start_code: |
        import numpy as np

        def rerank_doug_blog(query, bm25, **kwargs):
            docs = bm25(query, top_k=5)
            return [doc["id"] for doc in docs]
      edit:
        guards:
          - validation
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
        script = _commit_script(_bm25_patch())
        with _with_codegen_agent(script=script):
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

        assert result.artifact_path
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_path_continuation_fixture_e2e(_paths_root, tmp_path):
    try:
        source_path = Path("tests/fixtures/past_runs/20260502_025238")
        continue_from = tmp_path / "continued_run"
        shutil.copytree(source_path, continue_from)
        config_path = tmp_path / "codegen_continue_path.yml"
        config_path.write_text(
            f"""
strategy:
  name: codegen_continue_path_fixture
  type: codegen
  path: {continue_from}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - fielded_bm25
      edit:
        guards:
          - validation
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
        script = _commit_script(_fielded_bm25_patch())
        with _with_codegen_agent(script=script):
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

        assert result.artifact_path
        assert result.metadata["continued_from"] == str(Path(continue_from).expanduser())
        assert result.metadata["previous_rounds"] > 0
        assert result.metadata["rounds"] == result.metadata["previous_rounds"] + 1
        round_name = f"reranker_round_{result.metadata['rounds']}.py"
        round_path = Path(result.artifact_path) / round_name
        assert round_path.exists()
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_path_missing_creates_run_e2e(_paths_root, tmp_path):
    try:
        missing_path = tmp_path / "nope"
        config_path = tmp_path / "codegen_missing_path.yml"
        config_path.write_text(
            f"""
strategy:
  name: codegen_missing_path_fixture
  type: codegen
  path: {missing_path}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - fielded_bm25
      edit:
        guards:
          - validation
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
        with pytest.raises(FileNotFoundError, match="Training run path not found"):
            train_strategy(params)
    finally:
        _cleanup_temp_root()


@patch("exps.codegen.strategy.find_latest_codegen_run", lambda *_: None)
def test_codegen_without_trained_run_e2e(tmp_path):
    config_path = tmp_path / "codegen_no_run.yml"
    config_path.write_text(
        """
strategy:
  name: codegen_no_run_fixture
  type: codegen
  params:
    train:
      search_tools:
        - bm25
    run: {}
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
    with pytest.raises(ValueError, match="No trained codegen run found"):
        run_benchmark(params)


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_raw_tool_list_e2e(_paths_root, tmp_path):
    try:
        config_path = tmp_path / "codegen_raw_list.yml"
        config_path.write_text(
            """
strategy:
  name: codegen_raw_list_fixture
  type: codegen
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - raw:
            - get_corpus
      edit:
        guards:
          - validation
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
        script = _commit_script(_get_corpus_patch())
        with _with_codegen_agent(script=script):
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

        assert result.artifact_path
    finally:
        _cleanup_temp_root()


@pytest.mark.skip(reason="Flaky/slow in CI; embedding step times out")
def test_codegen_guarded_wands_ndcg_nonzero_e2e(tmp_path: Path):
    run_path = tmp_path / "codegen_guarded_wands"
    run_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "codegen_guarded_wands_small.yml"
    config_path.write_text(
        f"""
strategy:
  name: codegen_guarded_wands_small
  type: codegen
  path: {run_path}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - fielded_bm25
        - e5_base_v2
      edit:
        guards:
          - validation
          - length:
              max_lines: 10
              max_cols: 120
      eval:
        train_fraction: 0.20
        seed: 1234
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.
    run:
      top_k: 10
""".lstrip(),
        encoding="utf-8",
    )
    params = TrainParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="wands",
        num_queries=2,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    rounds = _load_rounds(Path(result.artifact_path))
    assert rounds[0]["mean_ndcg"] > 0.0


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_start_code_rerank_only_wrapper_e2e(_paths_root, tmp_path: Path):
    try:
        run_path = tmp_path / "codegen_rerank_only"
        run_path.mkdir(parents=True, exist_ok=True)
        config_path = tmp_path / "codegen_start_code_rerank_only_path.yml"
        config_path.write_text(
            f"""
strategy:
  name: codegen_start_code_rerank_only_fixture
  type: codegen
  path: {run_path}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      rounds: 0
      refresh_every: 1
      search_tools:
        - bm25
      start_code: |
        def rerank_doug_blog(query, bm25, **kwargs):
            docs = bm25(query, top_k=5)
            return [doc["id"] for doc in docs]
      edit:
        guards:
          - length
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
        params = TrainParams(
            strategy_path=str(config_path),
            base_path=None,
            dataset="doug_blog",
            num_queries=1,
            seed=123,
            workers=1,
            device=None,
            rounds=0,
        )
        result = train_strategy(params)

        reranker_path = Path(result.artifact_path) / "reranker.py"
        content = reranker_path.read_text(encoding="utf-8")
        assert Path(result.artifact_path) == run_path
        assert "def reranker(" in content
        assert "def rerank_doug_blog(" in content
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_path_uses_start_code_e2e(_paths_root, tmp_path: Path):
    try:
        run_path = tmp_path / "codegen_start_code_marker"
        run_path.mkdir(parents=True, exist_ok=True)
        config_path = tmp_path / "codegen_start_code_path_marker.yml"
        config_path.write_text(
            f"""
strategy:
  name: codegen_start_code_path_marker_fixture
  type: codegen
  path: {run_path}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      rounds: 0
      refresh_every: 1
      search_tools:
        - get_corpus
      start_code: |
        START_CODE_SENTINEL = True

        def reranker(query, top_k, get_corpus, **kwargs):
            corpus = get_corpus()
            return [str(doc_id) for doc_id in corpus.head(top_k)["doc_id"].tolist()]
      edit:
        guards:
          - length
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
        params = TrainParams(
            strategy_path=str(config_path),
            base_path=None,
            dataset="doug_blog",
            num_queries=1,
            seed=123,
            workers=1,
            device=None,
            rounds=0,
        )
        result = train_strategy(params)

        reranker_path = Path(result.artifact_path) / "reranker.py"
        content = reranker_path.read_text(encoding="utf-8")
        assert Path(result.artifact_path) == run_path
        assert "START_CODE_SENTINEL = True" in content
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_validation_guard_toggle_e2e(_paths_root, tmp_path: Path):
    try:
        run_path_on = tmp_path / "codegen_validation_on"
        run_path_on.mkdir(parents=True, exist_ok=True)
        config_path_on = tmp_path / "codegen_validation_on.yml"
        config_path_on.write_text(
            f"""
strategy:
  name: codegen_validation_on_fixture
  type: codegen
  path: {run_path_on}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      rounds: 0
      refresh_every: 1
      search_tools:
        - bm25
      edit:
        guards:
          - validation
      eval:
        train_fraction: 0.5
        seed: 123
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.
    run:
      top_k: 5
""".lstrip(),
            encoding="utf-8",
        )
        params_on = TrainParams(
            strategy_path=str(config_path_on),
            base_path=None,
            dataset="doug_blog",
            num_queries=2,
            seed=123,
            workers=1,
            device=None,
            rounds=0,
        )
        result_on = train_strategy(params_on)
        metadata_on = json.loads(
            (Path(result_on.artifact_path) / "metadata.json").read_text(encoding="utf-8")
        )
        rounds_on = _load_rounds(Path(result_on.artifact_path))
        assert metadata_on["num_validation_queries"] > 0
        assert rounds_on[0]["validation_query_count"] > 0

        run_path_off = tmp_path / "codegen_validation_off"
        run_path_off.mkdir(parents=True, exist_ok=True)
        config_path_off = tmp_path / "codegen_validation_off.yml"
        config_path_off.write_text(
            f"""
strategy:
  name: codegen_validation_off_fixture
  type: codegen
  path: {run_path_off}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      rounds: 0
      refresh_every: 1
      search_tools:
        - bm25
      edit:
        guards: []
      eval:
        train_fraction: 0.5
        seed: 123
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.
    run:
      top_k: 5
""".lstrip(),
            encoding="utf-8",
        )
        params_off = TrainParams(
            strategy_path=str(config_path_off),
            base_path=None,
            dataset="doug_blog",
            num_queries=2,
            seed=123,
            workers=1,
            device=None,
            rounds=0,
        )
        result_off = train_strategy(params_off)
        metadata_off = json.loads(
            (Path(result_off.artifact_path) / "metadata.json").read_text(encoding="utf-8")
        )
        rounds_off = _load_rounds(Path(result_off.artifact_path))
        assert metadata_off["num_validation_queries"] == 0
        assert rounds_off[0]["validation_query_count"] == 0
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_raw_tool_list_runner_e2e(_paths_root, tmp_path: Path):
    try:
        run_path = tmp_path / "codegen_raw_tool_list"
        run_path.mkdir(parents=True, exist_ok=True)
        config_path = tmp_path / "codegen_raw_tool_list.yml"
        config_path.write_text(
            f"""
strategy:
  name: codegen_raw_tool_list_fixture
  type: codegen
  path: {run_path}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      rounds: 0
      refresh_every: 1
      search_tools:
        - raw:
            - get_corpus
      start_code: |
        def rerank_doug_blog(query, get_corpus, **kwargs):
            corpus = get_corpus()
            top_k = int(kwargs.get("top_k", 5))
            return [str(doc_id) for doc_id in corpus.head(top_k)["doc_id"].tolist()]
      edit:
        guards:
          - length
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
        train_params = TrainParams(
            strategy_path=str(config_path),
            base_path=None,
            dataset="doug_blog",
            num_queries=1,
            seed=123,
            workers=1,
            device=None,
            rounds=0,
        )
        train_strategy(train_params)

        run_params = RunParams(
            strategy_path=str(config_path),
            base_path=None,
            dataset="doug_blog",
            num_queries=1,
            seed=123,
            workers=1,
            device=None,
            no_cache=True,
        )
        result = run_benchmark(run_params)

        assert result.metric_series is not None
        assert not result.metric_series.empty
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_minimal_round_state_e2e(_paths_root, tmp_path: Path):
    try:
        run_path = tmp_path / "codegen_run"
        run_path.mkdir()
        config_path = tmp_path / "codegen_minimal.yml"
        config_path.write_text(
            f"""
strategy:
  name: codegen_minimal_fixture
  type: codegen
  path: {run_path}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      rounds: 0
      refresh_every: 1
      search_tools:
        - get_corpus
      start_code: |
        def rerank_doug_blog(query, get_corpus, **kwargs):
            corpus = get_corpus()
            top_k = int(kwargs.get("top_k", 5))
            return [str(doc_id) for doc_id in corpus.head(top_k)["doc_id"].tolist()]
      edit:
        guards:
          - validation
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

        params = TrainParams(
            strategy_path=str(config_path),
            base_path=None,
            dataset="doug_blog",
            num_queries=1,
            seed=123,
            workers=1,
            device=None,
        )

        result = train_strategy(params)

        assert result.strategy_name == "codegen_minimal_fixture"
        assert Path(result.artifact_path).exists()
    finally:
        _cleanup_temp_root()


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
def test_codegen_raw_bm25_get_corpus_e2e(_paths_root, tmp_path: Path):
    try:
        codegen_dir = tmp_path / "codegen_raw_bm25"
        codegen_dir.mkdir()
        reranker_path = codegen_dir / "reranker.py"
        reranker_path.write_text(
            """
import numpy as np


def rerank_doug_blog(query, get_corpus, **kwargs):
    corpus = get_corpus()
    snowball = corpus["description_snowball"].array
    tokenizer = snowball.tokenizer
    terms = [term for term in tokenizer(query) if term]
    if not terms:
        return []

    doc_lengths = snowball.doclengths()
    if len(doc_lengths) == 0:
        return []
    avg_dl = float(doc_lengths.mean())
    if avg_dl <= 0:
        return []

    k1 = 0.6
    b = 0.62
    n_docs = len(corpus)
    scores = np.zeros(n_docs)

    for term in terms:
        term_freqs = snowball.termfreqs(term)
        doc_freq = snowball.docfreq(term)
        if doc_freq == 0:
            continue
        idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        denom = term_freqs + k1 * (1.0 - b + b * (doc_lengths / avg_dl))
        scores += idf * (term_freqs * (k1 + 1.0)) / np.where(denom == 0, 1.0, denom)

    top_k = int(kwargs.get("top_k", 10))
    if top_k <= 0:
        return []
    ranked = np.argsort(-scores)[:top_k]
    return [str(corpus.iloc[idx]["doc_id"]) for idx in ranked if scores[idx] > 0]
""".lstrip(),
            encoding="utf-8",
        )
        config_path = tmp_path / "codegen_raw_bm25.yml"
        config_path.write_text(
            f"""
strategy:
  name: codegen_raw_bm25_fixture
  type: codegen
  path: {codegen_dir}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - get_corpus
      edit:
        guards:
          - validation
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
        result = run_benchmark(params)

        assert result.metric_series is not None
        assert not result.metric_series.empty
    finally:
        _cleanup_temp_root()
