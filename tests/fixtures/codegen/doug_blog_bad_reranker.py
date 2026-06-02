# Bad ranker for doug_blog: only searches descriptions, so it misses title phrase matches
# that the judgments consider relevant.
def reranker(query, top_k, fielded_bm25, **kwargs):
    docs = fielded_bm25(query, fields=["description^99.0"], operator="or", top_k=top_k)
    return [str(doc["id"]) for doc in docs]
