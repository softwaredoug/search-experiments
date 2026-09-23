import numpy as np


def rerank_minimarco(query, get_corpus, **kwargs):
    np = __import__("numpy"); C = get_corpus(); A = C["description_snowball"].array; D = A.doclengths(); n = len(C)
    S=" a an and are as at be by do doe did from how in is it on or that the to use was what when where who why you "
    R = [t for t in A.tokenizer(query) if len(t) > 1]; T = [t for t in R if (" "+t+" ") not in S]
    if len(T) < 2: T = R
    k1 = 0.6; b = 0.7; m = float(D.mean()); sc = np.zeros(n); z = k1 * (1.0 - b + b * D / m)
    for t in T:
        tf = A.termfreqs(t); df = A.docfreq(t); idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)) if df else 0
        if df: sc += idf * (tf * (k1 + 1.0)) / np.where(tf + z == 0, 1.0, tf + z)
    return [str(C.iloc[i]["doc_id"]) for i in np.argsort(-sc)[:int(kwargs.get("top_k", 10))] if sc[i] > 0]
