def rerank_wands(*args, **kwargs):
    """
    Hybrid reranker for the wands dataset using reciprocal rank fusion (RRF).

    Accepts either positional or keyword inputs for:
      - query
      - fielded_bm25
      - search_embeddings
    """
    # Unpack inputs robustly to avoid duplicate 'query' kwargs/positional conflicts
    query = kwargs.get('query') if 'query' in kwargs else (args[0] if len(args) > 0 else '')
    fielded_bm25 = kwargs.get('fielded_bm25') if 'fielded_bm25' in kwargs else (args[1] if len(args) > 1 else None)
    search_embeddings = kwargs.get('search_embeddings') if 'search_embeddings' in kwargs else (args[2] if len(args) > 2 else None)

    # Defensive handling of empty/whitespace queries
    q = (query or "").strip()
    if not q:
        docs = fielded_bm25("", field_to_search='title', operator='or', top_k=10)
        return [str(doc.get('id')) for doc in docs]

    # Helper to safely call tools
    def safe_bm25(keywords, field_to_search, operator, k):
        try:
            return fielded_bm25(keywords, field_to_search=field_to_search, operator=operator, top_k=k) or []
        except Exception:
            return []

    def safe_embed(question, k):
        try:
            return search_embeddings(question, top_k=k) or []
        except Exception:
            return []

    # Retrieve from multiple views
    topk = int(kwargs.get('top_k', 20))
    topk = max(10, min(20, topk))  # keep within supported bounds

    bm25_title_or = safe_bm25(q, 'title', 'or', topk)
    bm25_title_and = safe_bm25(q, 'title', 'and', topk)
    bm25_desc_or = safe_bm25(q, 'description', 'or', topk)
    emb = safe_embed(q, topk)

    # Build rank maps
    def rank_map(results):
        r = {}
        for i, d in enumerate(results):
            did = str(d.get('id'))
            if did is None:
                continue
            # only first occurrence matters for rank-based fusion
            if did not in r:
                r[did] = i + 1  # 1-based rank
        return r

    r_title_or = rank_map(bm25_title_or)
    r_title_and = rank_map(bm25_title_and)
    r_desc_or = rank_map(bm25_desc_or)
    r_emb = rank_map(emb)

    # Weighted Reciprocal Rank Fusion
    k_rrf = 60
    w_title_or = 1.2
    w_title_and = 1.5
    w_desc_or = 0.9
    w_emb = 1.0

    # Collect all candidate ids
    candidate_ids = set(list(r_title_or.keys()) + list(r_title_and.keys()) + list(r_desc_or.keys()) + list(r_emb.keys()))

    # For lexical boosts we need quick access to titles/descriptions of at least one source
    # Build a metadata map from the first occurrence we find among sources
    meta = {}
    for src in (bm25_title_or, bm25_title_and, bm25_desc_or, emb):
        for d in src:
            did = str(d.get('id'))
            if did not in meta:
                meta[did] = {
                    'title': d.get('title', '') or '',
                    'description': d.get('description', '') or ''
                }

    q_lower = q.lower()
    q_terms = [t for t in q_lower.replace('"', ' ').replace("'", ' ').split() if t]
    q_terms_set = set(q_terms)

    scores = {}
    for did in candidate_ids:
        score = 0.0
        if did in r_title_or:
            score += w_title_or / (k_rrf + r_title_or[did])
        if did in r_title_and:
            score += w_title_and / (k_rrf + r_title_and[did])
        if did in r_desc_or:
            score += w_desc_or / (k_rrf + r_desc_or[did])
        if did in r_emb:
            score += w_emb / (k_rrf + r_emb[did])

        # Light lexical boost for strong title coverage
        m = meta.get(did, {})
        title = (m.get('title') or '').lower()
        if title:
            # Exact phrase presence boost
            if q_lower in title:
                score += 0.5
            # Token coverage boost when all query terms appear in title
            if q_terms and all(t in title for t in q_terms):
                score += 0.8
            else:
                # partial overlap boost proportional to coverage
                if q_terms:
                    overlap = sum(1 for t in q_terms_set if t in title)
                    score += 0.1 * (overlap / max(1, len(q_terms_set)))
        else:
            # fallback: check description lightly
            desc = (m.get('description') or '').lower()
            if desc and q_lower in desc:
                score += 0.2

        scores[did] = score

    # Sort by fused score desc, tiebreak on best individual rank
    def best_rank(did):
        ranks = [r.get(did, 10**9) for r in (r_title_or, r_title_and, r_desc_or, r_emb)]
        return min(ranks)

    ranked = sorted(scores.items(), key=lambda x: (-x[1], best_rank(x[0])))

    # Return up to 10 results as strings
    return [did for did, _ in ranked[:10]]
