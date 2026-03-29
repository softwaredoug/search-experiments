import numpy as np
from searcharray import SearchArray

from cheat_at_search import wands_data
from cheat_at_search.tokenizers import snowball_tokenizer

from prf.strategies.bm25_vect import bm25_vect


def test_bm25_vect_wands_titles_non_empty():
    corpus = wands_data.corpus.head(200)
    indexed = SearchArray.index(corpus["title"], snowball_tokenizer)
    matches = np.zeros(len(indexed), dtype=bool)
    matches[:10] = True

    vector = bm25_vect(indexed, matches)

    assert len(vector) > 0
    assert any(score > 0 for score in vector.values())
