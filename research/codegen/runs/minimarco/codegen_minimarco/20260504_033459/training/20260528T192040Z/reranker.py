def rerank_minimarco(query, fielded_bm25, get_corpus, **kwargs):
    return []
def reranker(query, top_k, *tool_fns, _r=rerank_minimarco, **kwargs):
    return _r(query,*tool_fns,top_k=top_k,**kwargs)
