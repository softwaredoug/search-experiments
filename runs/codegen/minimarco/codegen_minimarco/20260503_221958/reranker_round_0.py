import numpy as np
import pandas as pd


def rerank_minimarco(query, fielded_bm25, get_corpus, **kwargs):
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
