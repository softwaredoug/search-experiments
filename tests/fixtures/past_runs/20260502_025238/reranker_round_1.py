def rerank_wands(query, fielded_bm25, search_embeddings, **kwargs):
    b = fielded_bm25(query, fields=['title^9.3','description^4.1'], operator='or', top_k=20)
    e = search_embeddings(query, top_k=20)
    bs = [d['score'] for d in b]; es = [d['score'] for d in e]
    bmn,bmx = (min(bs),max(bs)) if bs else (0,0); emn,emx = (min(es),max(es)) if es else (0,0)
    s = {}
    for d in b: s[str(d['id'])] = 0.6*((d['score']-bmn)/(bmx-bmn+1e-9))
    for d in e: s[str(d['id'])] = s.get(str(d['id']),0)+0.4*((d['score']-emn)/(emx-emn+1e-9))
    return [k for k,_ in sorted(s.items(), key=lambda x:x[1], reverse=True)][:10]
