def rerank_wands(query, search, query_rewrite, **kwargs):
    docs=search(query, top_k=10)
    qt=query.lower(); toks=[t for t in qt.replace('-',' ').replace('/',' ').split() if t]
    def keyf(d):
        t=(d.get('title','') or '').lower(); s=(d.get('description','') or '').lower(); ts=t+' '+s
        return (qt in t, all(tok in ts for tok in toks), sum(tok in t for tok in toks), qt in s, (-t.find(qt) if qt in t else -1))
    docs=sorted(docs, key=keyf, reverse=True)
    return [str(d['id']) for d in docs]
