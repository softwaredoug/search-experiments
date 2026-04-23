import numpy as np
from searcharray import SearchArray

from cheat_at_search import wands_data
from cheat_at_search.tokenizers import snowball_tokenizer

from exps.strategies.prf_rerank_terms import compute_bm25_matrix


def test_compute_bm25_matrix_wands_description_sanity():
    corpus = wands_data.corpus.head(100)
    indexed = SearchArray.index(corpus["description"], snowball_tokenizer)

    doc_weights = np.zeros(len(indexed), dtype=float)
    doc_weights[:10] = 1.0

    matrix = compute_bm25_matrix(indexed, doc_weights, top_docs=10)

    assert matrix

    checked = 0
    for term, entries in matrix.items():
        doc_id, weight = entries[0]
        assert weight > 0
        assert indexed.termfreqs(term)[doc_id] > 0
        checked += 1
        if checked >= 5:
            break
