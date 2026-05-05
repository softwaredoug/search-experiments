def rerank_wands(query, search, query_rewrite, **kwargs):
    docs=search(query, top_k=20); qt=query.lower()
    toks=[t for t in qt.replace('-',' ').replace('/',' ').split() if t]
    alts=(query_rewrite(qt, max_alternatives=5).get('rewriters') or [qt])
    def keyf(d):
        t=(d.get('title','') or '').lower(); s=(d.get('description','') or '').lower(); ts=t+' '+s
        full=any(all(w in ts for w in a.replace('-',' ').replace('/',' ').split()) for a in alts)
        return (t.startswith(qt), qt in t, full, sum(tok in t for tok in toks), qt in s, (-t.find(qt) if qt in t else -1))
    docs=sorted(docs, key=keyf, reverse=True); return [str(d['id']) for d in docs]

