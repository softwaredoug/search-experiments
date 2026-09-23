def rerank_wands(query, fielded_bm25, search_embeddings, **kwargs):
    docs = fielded_bm25(query, fields=['title^9.3', 'description^4.1'], operator='or', top_k=10)
    return [str(doc['id']) for doc in docs]
