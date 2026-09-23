def rerank_wands(query, fielded_bm25, search_embeddings, query_rewrite, **kwargs):
    q=(query or '').strip(); alts=query_rewrite(q, max_alternatives=3).get('rewriters',[]) if q else []
    if alts: q+=' ' + alts[0]; op='and' if len(q.split())<=3 else 'or'
    em=search_embeddings(q, top_k=40); E={str(d['id']) for d in em[:7]}
    bm=fielded_bm25(q, fields=['title^9.3','description^4.1'], operator=op, top_k=40, k1=1.1, b=0.55)
    mb=max([d['score'] for d in bm] or [1.0]) or 1.0; me=max([d['score'] for d in em] or [1.0]) or 1.0
    S={}; [S.__setitem__(str(d['id']), S.get(str(d['id']),0)+0.58*d['score']/mb) for d in bm]
    [S.__setitem__(str(d['id']), S.get(str(d['id']),0)+0.42*d['score']/me) for d in em]
    [S.__setitem__(k, S.get(k,0)+0.25) for k in E if k in S]
    return [k for k,_ in sorted(S.items(), key=lambda x: x[1], reverse=True)][:10]
