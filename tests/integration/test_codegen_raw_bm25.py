from exps.runners.run import RunParams, run_benchmark


def test_run_codegen_raw_bm25_get_corpus(tmp_path):
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
        train_query_fraction: 0.2
        validation_query_fraction: 0.2
        training_seed: 123
        validation_seed: 456
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.
    run:
      top_k: 5
      path: {codegen_dir}
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
