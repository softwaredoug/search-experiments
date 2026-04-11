import numpy as np
from searcharray import SearchArray

from cheat_at_search import wands_data
from cheat_at_search.tokenizers import snowball_tokenizer

from prf.strategies.prf_rerank_terms import top_n_term_strengths


def test_prf_rerank_terms_wands_titles_non_empty():
    corpus = wands_data.corpus.head(200)
    indexed = SearchArray.index(corpus["title"], snowball_tokenizer)
    matches = np.zeros(len(indexed), dtype=bool)
    matches[:10] = True

    doc_weights = matches.astype(float)
    terms, scores, _, _ = top_n_term_strengths(
        indexed.array,
        doc_weights,
        top_docs=10,
        top_terms=5,
    )

    assert len(terms) > 0
    assert any(np.any(term_scores > 0) for term_scores in scores)
