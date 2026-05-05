def rerank_minimarco(query, fielded_bm25, get_corpus, **kwargs):
    import numpy as np; c=get_corpus(); a=c["description_snowball"].array; toks=[t for t in a.tokenizer(query) if t]
    sw=set(("what is are was were a an the in to for do doe did can you i me there where when who why how "
            "consid achiev and or on with from that").split())
    q=[t for t in toks if t not in sw]; toks=q if len(toks)>3 and len(q)>1 and "de" not in q else toks
    dl=a.doclengths(); n=len(c); avg=float(dl.mean()); s=np.zeros(n); K=.6*(.08+.92*dl/avg)
    for t in toks:
        tf=a.termfreqs(t); df=a.docfreq(t); s+=np.log(1+(n-df+.5)/(df+.5))*(tf*1.6)/np.maximum(tf+K,1e-9)
    s+=sum((.08*a.termfreqs(toks[i:i+2]) for i in range(len(toks)-1)),np.zeros(n))+(.6*a.termfreqs(toks) if len(toks)>2 else 0)
    return [str(c.iloc[i]["doc_id"]) for i in np.argsort(-s)[:int(kwargs.get("top_k",10))] if s[i]>0]
