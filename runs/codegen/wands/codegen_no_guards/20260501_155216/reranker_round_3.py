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

    # Lightweight query normalization/augmentation
    def normalize(qs: str):
        s = (qs or '').strip()
        s = s.replace('\u00d7', ' x ').replace('×', ' x ')
        # space-pad quotes then remove for better tokenization
        s = s.replace('"', ' ').replace("'", ' ')
        # collapse multiple spaces
        s = ' '.join(s.split())
        return s

    def expand_queries(qs: str):
        base = normalize(qs)
        variants = [base]
        # dimension normalization: "22 x 36" standardize spacing and also try without the x
        import re
        dim_pat = re.compile(r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)")
        if dim_pat.search(base):
            def repl(m):
                a, b = m.group(1), m.group(2)
                return f"{a} x {b}"
            std = dim_pat.sub(repl, base)
            if std not in variants:
                variants.append(std)
            # also try variant with just numbers separated (helps AND queries)
            just_nums = dim_pat.sub(lambda m: f"{m.group(1)} {m.group(2)}", base)
            if just_nums not in variants:
                variants.append(just_nums)
        # color spelling expansion
        low = base.lower()
        if 'grey' in low and 'gray' not in low:
            variants.append(base + ' gray')
        if 'gray' in low and 'grey' not in low:
            variants.append(base + ' grey')
        # remove common punctuation/hyphens for a loose variant
        loose = base.replace('-', ' ')
        loose = ' '.join(loose.split())
        if loose and loose not in variants:
            variants.append(loose)
        return variants[:4]  # cap to avoid explosion

    variants = expand_queries(q)

    # Detect query characteristics for adaptive weighting/boosting
    import re
    q_lower = variants[0].lower()
    q_terms = [t for t in q_lower.split() if t]
    q_terms_set = set(q_terms)
    q_nums = re.findall(r"\d+(?:\.\d+)?", q_lower)
    has_dims = len(q_nums) >= 2

    # Retrieve from multiple views
    topk = int(kwargs.get('top_k', 20))
    topk = max(10, min(20, topk))  # keep within supported bounds

    # Core sources using base query
    bm25_title_or = safe_bm25(variants[0], 'title', 'or', topk)
    bm25_title_and = safe_bm25(variants[0], 'title', 'and', topk)
    bm25_desc_or = safe_bm25(variants[0], 'description', 'or', topk)
    bm25_desc_and = safe_bm25(variants[0], 'description', 'and', topk)

    # Variant sources (prefer AND for precision)
    var_title_ands = []
    for v in variants[1:]:
        var_title_ands.append(safe_bm25(v, 'title', 'and', topk))

    emb = safe_embed(variants[0], topk)

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
    r_desc_and = rank_map(bm25_desc_and)
    r_var_title_ands = [rank_map(v) for v in var_title_ands]
    r_emb = rank_map(emb)

    # Weighted Reciprocal Rank Fusion
    k_rrf = 60
    w_title_or = 1.2
    w_title_and = 1.8 if has_dims else 1.5
    w_desc_or = 0.9
    w_desc_and = 1.1
    w_var_title_and = 1.4 if has_dims else 1.2
    w_emb = 1.0

    # Collect all candidate ids
    candidate_ids = set(list(r_title_or.keys()) + list(r_title_and.keys()) + list(r_desc_or.keys()) + list(r_desc_and.keys()) + list(r_emb.keys()))
    for rm in r_var_title_ands:
        candidate_ids.update(rm.keys())

    # For lexical boosts we need quick access to titles/descriptions/scores of at least one source
    # Build a metadata map from the first occurrence we find among sources
    meta = {}
    def harvest(src_list):
        for src in src_list:
            for d in src:
                did = str(d.get('id'))
                if did not in meta:
                    meta[did] = {
                        'title': d.get('title', '') or '',
                        'description': d.get('description', '') or '',
                        'score': d.get('score', None)
                    }
    harvest([bm25_title_or, bm25_title_and, bm25_desc_or, bm25_desc_and, emb] + var_title_ands)

    # Color-aware adjustments
    color_terms = {"black","white","gray","grey","blue","red","green","pink","purple","gold","silver","bronze","brass","brown","beige","ivory","charcoal","champagne"}
    q_colors = {t for t in q_terms_set if t in color_terms}

    scores = {}
    for did in candidate_ids:
        score = 0.0
        if did in r_title_or:
            score += w_title_or / (k_rrf + r_title_or[did])
        if did in r_title_and:
            score += w_title_and / (k_rrf + r_title_and[did])
        if did in r_desc_or:
            score += w_desc_or / (k_rrf + r_desc_or[did])
        if did in r_desc_and:
            score += w_desc_and / (k_rrf + r_desc_and[did])
        for rm in r_var_title_ands:
            if did in rm:
                score += w_var_title_and / (k_rrf + rm[did])
        if did in r_emb:
            score += w_emb / (k_rrf + r_emb[did])

        # Light lexical boosts
        m = meta.get(did, {})
        title = (m.get('title') or '').lower()
        desc = (m.get('description') or '').lower()

        if title:
            # Exact phrase presence boost for the base normalized query only
            if q_lower in title:
                score += 0.5
            # Token coverage boost when all base query terms appear in title
            if q_terms and all(t in title for t in q_terms):
                score += 0.8
            else:
                # partial overlap boost proportional to coverage
                if q_terms:
                    overlap = sum(1 for t in q_terms_set if t in title)
                    score += 0.1 * (overlap / max(1, len(q_terms_set)))
        else:
            if desc and q_lower in desc:
                score += 0.2

        # Number/dimension coverage boost (applies to title+desc)
        if has_dims:
            text = f"{title} {desc}"
            num_overlap = sum(1 for n in q_nums if n in text)
            if len(q_nums) and num_overlap == len(q_nums):
                score += 0.6  # strong boost when all numbers match
            elif num_overlap:
                score += 0.25 * (num_overlap / len(q_nums))

        # Prefer description-backed color matches to avoid brand-name collisions
        if q_colors:
            for c in q_colors:
                in_title = (f" {c} " in f" {title} ") or c in title
                in_desc = (f" {c} " in f" {desc} ") or c in desc
                if in_desc:
                    score += 0.25  # color mentioned in attributes/description
                elif in_title and not in_desc:
                    score -= 0.15  # likely brand or non-attribute usage

        # Small raw BM25 score nudging if available (helps break ties)
        raw = m.get('score')
        if isinstance(raw, (int, float)):
            score += 0.01 * (raw / 100.0)  # normalized tiny bump

        scores[did] = score

    # Sort by fused score desc, tiebreak on best individual rank
    def best_rank(did):
        ranks = [r.get(did, 10**9) for r in (r_title_or, r_title_and, r_desc_or, r_desc_and, r_emb)]
        for rm in r_var_title_ands:
            ranks.append(rm.get(did, 10**9))
        return min(ranks)

    ranked = sorted(scores.items(), key=lambda x: (-x[1], best_rank(x[0])))

    # Return up to 10 results as strings
    return [did for did, _ in ranked[:10]]
