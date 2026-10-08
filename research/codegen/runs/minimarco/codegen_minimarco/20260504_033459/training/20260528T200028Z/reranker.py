def reranker(query, top_k, fielded_bm25, get_corpus, **kwargs):
    r=fielded_bm25(query,["description^1"],"or",int(top_k),1.2,.75)
    return [str(x["id"]) for x in r]
