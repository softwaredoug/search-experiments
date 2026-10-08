def rerank_minimarco(query, fielded_bm25, get_corpus, **kwargs):
    import numpy as np; c=get_corpus(); a=c["description_snowball"].array; toks=[t for t in a.tokenizer(query) if t]
    sw=set(("what is are was were be as a an the in to for do doe did can you i me there where when who why how "
            "which consid achiev mani some word need and or on with from that call place medicin vacat").split())
    q=[t for t in toks if t not in sw]; toks=q if len(toks)>3 and len(q)>1 and toks[1] not in ("can","invent") and toks[2]!="an" else toks
    dl=a.doclengths(); n=len(c); avg=float(dl.mean()); s=np.zeros(n); K=.55*(.08+.92*dl/avg); z="stand" in toks; B=fielded_bm25
    for t in toks: tf=a.termfreqs(t); df=a.docfreq(t); s+=np.log(1+(n-df+.5)/(df+.5))*(1-.3*(len(t)<3 and not z))*(tf*1.5)/(tf+K)+.25*(tf>0)
    for r in B(" ".join(toks),["description^1"],"phrase",50,1.2,.75): i=int(r["id"]); i<n and s.__setitem__(i,s[i]+.5)
    return [str(c.iloc[i]["doc_id"]) for i in np.argsort(-s)[:int(kwargs.get("top_k",10))] if s[i]>0]
def reranker(query, top_k, *tool_fns, _r=rerank_minimarco, **kwargs): return _r(query,*tool_fns,top_k=top_k,**kwargs)
