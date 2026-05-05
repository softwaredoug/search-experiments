def rerank_wands(query, fielded_bm25, search_embeddings, **kwargs):
    b = fielded_bm25(query, fields=['title^10.5','description^3.9'], operator='or', top_k=20, k1=1.2, b=0.6)
    e = search_embeddings(query, top_k=20)
    s = {}; k = 50.0
    for i,d in enumerate(b): s[str(d['id'])] = s.get(str(d['id']),0.0)+0.45/(k+i+1.0)
    for i,d in enumerate(e): s[str(d['id'])] = s.get(str(d['id']),0.0)+0.55/(k+i+1.0)
    ib = {str(d['id']) for d in b}; ie = {str(d['id']) for d in e}
    for i in ib & ie: s[i] = s.get(i,0.0)+0.01
    return [k for k,_ in sorted(s.items(), key=lambda x:x[1], reverse=True)][:10]


def rerank_doug_blog(query, fielded_bm25, search_embeddings, **kwargs):
    return rerank_wands(query, fielded_bm25, search_embeddings, **kwargs)
