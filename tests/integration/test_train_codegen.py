from pathlib import Path

from exps.runners.train import TrainParams, train_strategy


def test_train_codegen_minimal_round_state(tmp_path: Path):
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
