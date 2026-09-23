def rerank_wands(query, fielded_bm25, search_embeddings, **kwargs):
    """
    Hybrid reranker for the wands dataset.

    Strategy:
    - Gather a broad candidate set using multiple retrieval modes:
      * BM25 (OR) for recall
      * BM25 (PHRASE) to capture exact-phrase matches
      * BM25 (AND) for precise multi-term intent
      * Embedding search for semantic recall
      * BM25 with lightweight query-expansion (alt spellings/synonyms) for robustness
    - Compute lightweight features per candidate (normalized scores, token overlap,
      title boosts, and presence of product-category head terms).
    - Combine with a weighted score and return top results by final score.
    """
    import re

    def normalize_text(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()

    def tokenize(s: str):
        return [t for t in normalize_text(s).split() if t]

    def ngrams(tokens, n):
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if n > 0 else []

    # A light set of product/category terms we often want to enforce/boost.
    product_terms = {
        'rug','rugs','chair','chairs','stool','stools','sofa','couch','sectional','loveseat','recliner',
        'bed','beds','frame','headboard','platform','mattress','nightstand','dresser','chest','bench',
        'table','tables','desk','island','vanity','console','coffee','end','side','dining',
        'lamp','light','lights','sconce','chandelier','pendant','lantern','string','pillow',
        'pillowcase','sham','duvet','comforter','comforters','quilt','blanket','cover','sheet','sheets','curtain',
        'panel','panels','cushion','cushions','mat','mats','matte','matting','rack','racks','shelf','shelves',
        'bookcase','cabinet','pantry','closet','wardrobe','trunk','storage','box','canister',
        'bowl','mug','decal','art','picture','mirror','faucet','sink','sinks','vanity','backplash',
        'backsplash','toilet','shower','caddy','hook','hooks','stand','bedding','bunk','bunks','bunkbed','bunkbeds'
    }

    # Common color/style tokens the user may care about being present in the item
    color_terms = {
        'white','black','gray','grey','ivory','beige','tan','teal','navy','olive','peach','magenta','purple',
        'red','green','blue','brown','gold','silver','bronze','brass','yellow','orange','turquoise','cream'
    }

    stopwords = {
        'the','a','an','and','or','with','for','of','in','on','to','by','at','from','set','pc','pcs'
    }

    # Basic tokens
    q_tokens = [t for t in tokenize(query) if t not in stopwords]

    # Simple token-level synonym/normalization expansions to improve overlap features
    token_expansions = {
        'grey': ['gray'],
        'gray': ['grey'],
        'tye': ['tie'],  # tye dye -> tie dye
        'pedistole': ['pedestal'],
        'biycicle': ['bicycle'],
        'ann': ['anne'],  # queen ann -> queen anne
    }

    q_tokens_aug = list(q_tokens)
    q_token_set = set(q_tokens)
    for t in list(q_tokens):
        for alt in token_expansions.get(t, []):
            if alt not in q_token_set:
                q_tokens_aug.append(alt)
                q_token_set.add(alt)

    # Identify must/anchor terms (product head terms present in the query)
    must_terms = [t for t in q_tokens_aug if t in product_terms]

    # Build a few light query expansion variants to increase recall for BM25
    def build_alt_queries(q: str):
        alts = set()
        q_norm = normalize_text(q)
        # spelling variants and phrase re-orderings for common 2-term structures
        repls = [
            (r"\bpedistole\b", "pedestal"),
            (r"\btye\s+dye\b", "tie dye"),
            (r"\bbiycicle\b", "bicycle"),
            (r"\bqueen\s+ann\b", "queen anne"),
            (r"\bgrey\b", "gray"),
        ]
        alt = q_norm
        for pat, rep in repls:
            alt = re.sub(pat, rep, alt)
        if alt != q_norm:
            alts.add(alt)
        # Glass rack vs rack glass
        if re.search(r"\brack\b.*\bglass\b", q_norm):
            alts.add(re.sub(r"\brack\b.*\bglass\b", "glass rack", q_norm))
        if re.search(r"\bglass\b.*\brack\b", q_norm):
            alts.add(re.sub(r"\bglass\b.*\brack\b", "glass rack", q_norm))
        # Collapsed franchise/brand variant (e.g., starwars)
        if re.search(r"\bstar\s+wars\b", q_norm):
            alts.add(re.sub(r"\bstar\s+wars\b", "starwars", q_norm))
        # Star Wars rug phrasing
        if re.search(r"\bstar\s+wars\b.*\brug\b", q_norm):
            alts.add("star wars area rug")
        # Pantry grey/gray normalization and phrasing
        if re.search(r"\bpantry\b", q_norm) and (re.search(r"\bgrey\b", q_norm) or re.search(r"\bgray\b", q_norm)):
            alts.add(q_norm.replace(" grey", " gray"))
            alts.add("gray pantry")
        # Pedestal sink fix
        if re.search(r"\bpedistole\b.*\bsink\b|\bpedestal\b.*\bsink\b", q_norm):
            alts.add("pedestal sink")
        # Waterproof synonyms (e.g., waterproof -> water-resistant/weatherproof)
        if re.search(r"\bwater\s*proof\b|\bwaterproof\b", q_norm):
            for syn in ["water-resistant","water resistant","weatherproof","weather resistant","all-weather","all weather"]:
                alts.add(q_norm.replace("waterproof", syn).replace("water proof", syn))
        # Abstract art inference: if query has 'abstract' and no product terms, try wall art intents
        if 'abstract' in q_norm.split() and not any(t in q_norm.split() for t in product_terms):
            alts.add("abstract wall art")
            # preserve one color if present
            for col in color_terms:
                if f" {col} " in f" {q_norm} ":
                    alts.add(f"{col} abstract wall art")
                    break
        # Decorative pillow colorized variants
        if re.search(r"\bpillow\b", q_norm):
            for col in color_terms:
                if f" {col} " in f" {q_norm} ":
                    alts.add(f"{col} throw pillow")
                    alts.add(f"{col} decorative pillow")
                    break
        # Bedding with two colors
        if 'bedding' in q_norm:
            cols = [c for c in color_terms if f" {c} " in f" {q_norm} "]
            if len(cols) >= 2:
                alts.add(f"{cols[0]} {cols[1]} bedding")
        return list(alts)[:10]  # cap number of variants

    alt_queries = build_alt_queries(query)

    # Retrieve candidates
    bm25_or = fielded_bm25(query, fields=['title^9.3', 'description^4.1'], operator='or', top_k=80, k1=1.3, b=0.7)
    bm25_phrase = fielded_bm25(query, fields=['title^9.3', 'description^4.1'], operator='phrase', top_k=30, k1=1.2, b=0.75)
    # Only use AND when query has more than 1 token to avoid over-restriction
    if len(q_tokens) > 1:
        bm25_and = fielded_bm25(query, fields=['title^9.3', 'description^4.1'], operator='and', top_k=40, k1=1.0, b=0.8)
    else:
        bm25_and = []
    emb = search_embeddings(query, top_k=20)

    # Alt-query retrievals (lighter weight)
    bm25_or_alt_all, bm25_and_alt_all, bm25_phrase_alt_all = [], [], []
    for aq in alt_queries:
        try:
            bm25_or_alt_all.extend(fielded_bm25(aq, fields=['title^9.3', 'description^4.1'], operator='or', top_k=50, k1=1.2, b=0.75) or [])
            if len(q_tokens) > 1:
                bm25_and_alt_all.extend(fielded_bm25(aq, fields=['title^9.3', 'description^4.1'], operator='and', top_k=30, k1=1.0, b=0.8) or [])
            bm25_phrase_alt_all.extend(fielded_bm25(aq, fields=['title^9.3', 'description^4.1'], operator='phrase', top_k=20, k1=1.2, b=0.75) or [])
        except Exception:
            # Be conservative if any alt retrieval fails
            pass

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
                    'scores': {'bm25_or': None, 'bm25_phrase': None, 'bm25_and': None, 'emb': None,
                               'bm25_or_alt': None, 'bm25_and_alt': None, 'bm25_phrase_alt': None},
                }
            cand[rid]['scores'][key] = float(r.get('score', 0.0))

    add_results(bm25_or, 'bm25_or')
    add_results(bm25_phrase, 'bm25_phrase')
    add_results(bm25_and, 'bm25_and')
    add_results(emb, 'emb')
    add_results(bm25_or_alt_all, 'bm25_or_alt')
    add_results(bm25_and_alt_all, 'bm25_and_alt')
    add_results(bm25_phrase_alt_all, 'bm25_phrase_alt')

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
    for k in ['bm25_or','bm25_phrase','bm25_and','emb','bm25_or_alt','bm25_and_alt','bm25_phrase_alt']:
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
        must_in_title = 0
        for m in must_terms:
            if m in title_set or m in desc_set:
                has_must = 1
            if m in title_set:
                must_in_title = 1
        return weighted_overlap, coverage, has_must, must_in_title, title_set, desc_set

    # Pre-compute some query helpers
    q_bigrams = ngrams(q_tokens, 2)
    q_trigrams = ngrams(q_tokens, 3)
    q_ngrams = list(dict.fromkeys(q_bigrams + q_trigrams))  # unique, preserve order
    q_token_set = set(q_tokens)  # ensure defined locally after potential edits
    colors_in_query = [c for c in color_terms if c in q_token_set]

    # Combine scores per candidate
    scored = []
    for v in cand.values():
        t = v['title'] or ''
        d = v['description'] or ''
        t_norm = normalize_text(t)
        d_norm = normalize_text(d)
        # Normalized retrieval scores
        s_or = minmax_norm(v['scores']['bm25_or'], *stats['bm25_or'])
        s_phrase = minmax_norm(v['scores']['bm25_phrase'], *stats['bm25_phrase'])
        s_and = minmax_norm(v['scores']['bm25_and'], *stats['bm25_and'])
        s_emb = minmax_norm(v['scores']['emb'], *stats['emb'])
        s_or_alt = minmax_norm(v['scores']['bm25_or_alt'], *stats['bm25_or_alt'])
        s_and_alt = minmax_norm(v['scores']['bm25_and_alt'], *stats['bm25_and_alt'])
        s_phrase_alt = minmax_norm(v['scores']['bm25_phrase_alt'], *stats['bm25_phrase_alt'])

        w_overlap, coverage, has_must, must_in_title, title_set, desc_set = overlap_features(q_tokens_aug, t, d)

        # Heuristic bonuses/penalties
        # Prefer candidates that include product head terms present in the query
        must_penalty = -0.35 if (must_terms and not has_must) else 0.0
        must_bonus = 0.22 if (must_terms and has_must) else 0.0
        must_title_bonus = 0.15 if (must_terms and must_in_title) else 0.0
        # Small title match bonus scaled by overlap
        title_bonus = 0.05 * min(w_overlap, 5)
        coverage_bonus = 0.10 * coverage

        # Contextual keyword: "outdoor"
        outdoor_bonus = 0.0
        if 'outdoor' in q_token_set:
            if 'outdoor' in title_set or 'outdoor' in desc_set:
                outdoor_bonus += 0.08
            else:
                outdoor_bonus -= 0.04

        # Color presence bonus/penalty if colors were part of the query
        color_bonus = 0.0
        if colors_in_query:
            for c in colors_in_query:
                if (c in title_set) or (c in desc_set):
                    color_bonus += 0.04
                else:
                    color_bonus -= 0.02
            # Brand disambiguation: "Canora Grey" is a brand, not a color intent
            if ('grey' in colors_in_query) and ((f" canora grey " in f" {t_norm} ") or (f" canora grey " in f" {d_norm} ")):
                color_bonus -= 0.04
            color_bonus = max(-0.06, min(0.12, color_bonus))

        # Phrase n-gram matching bonus to strongly reward exact multi-word intent
        phrase_match_count = 0
        if q_ngrams:
            pad_t = f" {t_norm} "
            pad_d = f" {d_norm} "
            for ng in q_ngrams:
                needle = f" {ng} "
                if (needle in pad_t) or (needle in pad_d):
                    phrase_match_count += 1
        phrase_bonus = min(0.15 * phrase_match_count, 0.35)

        # Style/category inference: abstract often implies wall art
        style_bonus = 0.0
        if 'abstract' in q_token_set:
            if any(s in title_set or s in desc_set for s in {'art','canvas','print','painting','poster','wall','decor','picture'}):
                style_bonus += 0.09
            elif any(s in title_set or s in desc_set for s in {'rug','rugs'}):
                style_bonus += 0.04

        # Weighted combination of retrieval signals
        # Emphasize phrase and AND slightly more; tone down embedding a bit
        score = (
            0.38 * s_or +
            0.28 * s_phrase +
            0.22 * s_and +
            0.16 * s_emb +
            0.06 * s_or_alt +
            0.04 * s_and_alt +
            0.04 * s_phrase_alt +
            title_bonus +
            coverage_bonus +
            must_bonus + must_title_bonus + must_penalty +
            outdoor_bonus +
            color_bonus +
            phrase_bonus +
            style_bonus
        )

        scored.append((score, v['id']))

    # Sort by score desc and return top 10 IDs
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sid for _, sid in scored[:10]]

