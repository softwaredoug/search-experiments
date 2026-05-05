def rerank_wands(query, fielded_bm25, search_embeddings, **kwargs):
    """
    Hybrid reranker for the wands dataset.

    Strategy:
    - Gather a broad candidate set using multiple retrieval modes:
      * BM25 (OR) for recall
      * BM25 (PHRASE) to capture exact-phrase matches
      * BM25 (AND) for precise multi-term intent
      * Embedding search for semantic recall
    - Compute lightweight features per candidate (normalized scores, token overlap,
      title boosts, and presence of product-category head terms).
    - Combine with a weighted score and return top results by final score.
    """
    import math
    import re

    def normalize_text(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()

    def tokenize(s: str):
        return [t for t in normalize_text(s).split() if t]

    # A light set of product/category terms we often want to enforce/boost.
    product_terms = {
        'rug','chair','chairs','stool','stools','sofa','couch','sectional','loveseat','recliner',
        'bed','frame','headboard','platform','mattress','nightstand','dresser','chest','bench',
        'table','tables','desk','island','vanity','console','coffee','end','side','dining',
        'lamp','light','lights','sconce','chandelier','pendant','lantern','string','pillow',
        'pillowcase','sham','duvet','comforter','quilt','blanket','cover','sheet','curtain',
        'panel','panels','cushion','cushions','mat','matte','matting','rack','shelf','shelves',
        'bookcase','cabinet','pantry','closet','wardrobe','trunk','storage','box','canister',
        'bowl','mug','decal','art','picture','mirror','faucet','sink','vanity','backplash',
        'backsplash','toilet','shower','caddy','hook','hooks','stand'
    }

    stopwords = {
        'the','a','an','and','or','with','for','of','in','on','to','by','at','from','set','pc','pcs'
    }

    q_tokens = [t for t in tokenize(query) if t not in stopwords]
    q_text_norm = " ".join(q_tokens)

    # Identify must/anchor terms (product head terms present in the query)
    must_terms = [t for t in q_tokens if t in product_terms]

    # Retrieve candidates
    bm25_or = fielded_bm25(query, fields=['title^9.3', 'description^4.1'], operator='or', top_k=80, k1=1.3, b=0.7)
    bm25_phrase = fielded_bm25(query, fields=['title^9.3', 'description^4.1'], operator='phrase', top_k=30, k1=1.2, b=0.75)
    # Only use AND when query has more than 1 token to avoid over-restriction
    if len(q_tokens) > 1:
        bm25_and = fielded_bm25(query, fields=['title^9.3', 'description^4.1'], operator='and', top_k=40, k1=1.0, b=0.8)
    else:
        bm25_and = []
    emb = search_embeddings(query, top_k=20)

    # Collect candidates
    cand = {}
    def add_results(results, key):
        for r in results:
            rid = str(r['id'])
            if rid not in cand:
                cand[rid] = {
                    'id': rid,
                    'title': r.get('title','') or '',
                    'description': r.get('description','') or '',
                    'scores': {'bm25_or': None, 'bm25_phrase': None, 'bm25_and': None, 'emb': None},
                }
            cand[rid]['scores'][key] = float(r.get('score', 0.0))

    add_results(bm25_or, 'bm25_or')
    add_results(bm25_phrase, 'bm25_phrase')
    add_results(bm25_and, 'bm25_and')
    add_results(emb, 'emb')

    if not cand:
        return []

    # Gather raw score arrays for normalization
    def collect_scores(key):
        vals = [v['scores'][key] for v in cand.values() if v['scores'][key] is not None]
        return vals

    def minmax_norm(x, lo, hi):
        if x is None or lo is None or hi is None or hi <= lo:
            return 0.0
        return (x - lo) / (hi - lo)

    stats = {}
    for k in ['bm25_or','bm25_phrase','bm25_and','emb']:
        arr = collect_scores(k)
        if arr:
            stats[k] = (min(arr), max(arr))
        else:
            stats[k] = (None, None)

    def overlap_features(qtoks, title, desc):
        title_toks = tokenize(title)
        desc_toks = tokenize(desc)
        title_set, desc_set = set(title_toks), set(desc_toks)
        qset = set(qtoks)
        # Overlap counts
        title_overlap = len(qset & title_set)
        desc_overlap = len(qset & desc_set)
        # Weighted overlap favors title
        weighted_overlap = title_overlap * 1.5 + desc_overlap * 1.0
        # Fractional coverage of query terms
        coverage = (title_overlap + desc_overlap) / max(1, len(qset))
        # Presence of any must term in doc text
        has_must = 0
        for m in must_terms:
            if m in title_set or m in desc_set:
                has_must = 1
                break
        return weighted_overlap, coverage, has_must

    # Combine scores per candidate
    scored = []
    for v in cand.values():
        t = v['title'] or ''
        d = v['description'] or ''
        # Normalized retrieval scores
        s_or = minmax_norm(v['scores']['bm25_or'], *stats['bm25_or'])
        s_phrase = minmax_norm(v['scores']['bm25_phrase'], *stats['bm25_phrase'])
        s_and = minmax_norm(v['scores']['bm25_and'], *stats['bm25_and'])
        s_emb = minmax_norm(v['scores']['emb'], *stats['emb'])

        w_overlap, coverage, has_must = overlap_features(q_tokens, t, d)

        # Heuristic bonuses/penalties
        # Strongly prefer candidates that include product head terms present in the query
        must_bonus = 0.25 if (must_terms and has_must) else ( -0.25 if must_terms and not has_must else 0.0 )
        # Small title match bonus scaled by overlap
        title_bonus = 0.05 * min(w_overlap, 5)
        coverage_bonus = 0.10 * coverage

        # Weighted combination of retrieval signals
        # Phrase gets a decent boost when available; emb helps semantic gaps
        score = (
            0.45 * s_or +
            0.20 * s_phrase +
            0.15 * s_and +
            0.20 * s_emb +
            title_bonus +
            coverage_bonus +
            must_bonus
        )

        scored.append((score, v['id']))

    # Sort by score desc and return top 10 IDs
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sid for _, sid in scored[:10]]
