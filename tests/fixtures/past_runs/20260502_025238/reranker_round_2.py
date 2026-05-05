def rerank_wands(query, fielded_bm25, search_embeddings, **kwargs):
    b = fielded_bm25(query, fields=['title^9.3','description^4.1'], operator='or', top_k=20)
    e = search_embeddings(query, top_k=20)
    s = {}; k = 60.0
    for i,d in enumerate(b): s[str(d['id'])] = s.get(str(d['id']),0.0)+0.7/(k+i+1.0)
    for i,d in enumerate(e): s[str(d['id'])] = s.get(str(d['id']),0.0)+0.3/(k+i+1.0)
    return [k for k,_ in sorted(s.items(), key=lambda x:x[1], reverse=True)][:10]
