# Good ranker for doug_blog: searches title phrases that the judgments treat as relevant.
def reranker(query, top_k, fielded_bm25, **kwargs):
    docs = fielded_bm25(query, fields=["title^99.0"], operator="or", top_k=top_k)
    return [str(doc["id"]) for doc in docs]
