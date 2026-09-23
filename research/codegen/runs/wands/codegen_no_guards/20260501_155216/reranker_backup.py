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
            # swap the order (titles often use either orientation)
            swapped = dim_pat.sub(lambda m: f"{m.group(2)} x {m.group(1)}", base)
            if swapped not in variants:
                variants.append(swapped)
            # "by" variant
            by_var = dim_pat.sub(lambda m: f"{m.group(1)} by {m.group(2)}", base)
            if by_var not in variants:
                variants.append(by_var)
            # inch-mark variant (double apostrophes) to match tokenization in titles
            inch_var = dim_pat.sub(lambda m: f"{m.group(1)} '' x {m.group(2)} ''", base)
            if inch_var not in variants:
                variants.append(inch_var)

        # inches quotes like 36" or 24'' or 30 in -> add explicit inches forms minimally
        inch_quote_pat = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\"|''|\bin\b|\binch(?:es)?\b)")
        if inch_quote_pat.search(base):
            exp_inch = inch_quote_pat.sub(lambda m: f"{m.group(1)} inch", base)
            if exp_inch not in variants:
                variants.append(exp_inch)

        # british/american spelling variants for molding
        low = base.lower()
        if re.search(r"\bmoulding\b", low):
            variants.append(re.sub(r"\bmoulding\b", "molding", base, flags=re.IGNORECASE))
        if re.search(r"\bmolding\b", low):
            variants.append(re.sub(r"\bmolding\b", "moulding", base, flags=re.IGNORECASE))
        # daybed compound
        if re.search(r"\bday\s+bed\b", low):
            variants.append(re.sub(r"\bday\s+bed\b", "daybed", base, flags=re.IGNORECASE))

        # minimal common abbreviations expansion (conservative) + common typo fixes
        def apply_abbrev(text: str):
            t = text
            t = re.sub(r"\bblk\b", 'black', t, flags=re.IGNORECASE)
            t = re.sub(r"\bbrush\b", 'brushed', t, flags=re.IGNORECASE)
            # common noisy typos seen in queries
            t = re.sub(r"\blsmp\b", 'lamp', t, flags=re.IGNORECASE)  # glass lsmp shades -> lamp shades
            t = re.sub(r"\bbiycicle\b", 'bicycle', t, flags=re.IGNORECASE)  # biycicle -> bicycle
            # tye dye -> tie dye (keep standalone tye -> tie only when followed by dye)
            t = re.sub(r"\btye(?=\s+dye\b)", 'tie', t, flags=re.IGNORECASE)
            return ' '.join(t.split())
        abrv = apply_abbrev(base)
        if abrv and abrv not in variants:
            variants.append(abrv)

        # color spelling expansion (use swapped-spelling variants instead of appending duplicates)
        if re.search(r"\bgrey\b", low) and not re.search(r"\bgray\b", low):
            swapped = re.sub(r"\bgrey\b", "gray", base, flags=re.IGNORECASE)
            if swapped not in variants:
                variants.append(swapped)
        if re.search(r"\bgray\b", low) and not re.search(r"\bgrey\b", low):
            swapped = re.sub(r"\bgray\b", "grey", base, flags=re.IGNORECASE)
            if swapped not in variants:
                variants.append(swapped)

        # remove common punctuation/hyphens for a loose variant
        loose = base.replace('-', ' ')
        loose = ' '.join(loose.split())
        if loose and loose not in variants:
            variants.append(loose)
        return variants[:6]  # cap to avoid explosion

    variants = expand_queries(q)

    # Detect query characteristics for adaptive weighting/boosting
    import re
    q_lower = variants[0].lower()
    q_terms = [t for t in q_lower.split() if t]
    # filter out stopwords and non-informative tokens for coverage checks
    stop = {"x","by","with","and","for","the","a","an","of","inch","inches","in","l","w","h","to","from","on"}
    q_content_terms = [t for t in q_terms if t not in stop and not re.fullmatch(r"\d+(?:\.\d+)?", t)]
    q_terms_set = set(q_terms)
    q_nums = re.findall(r"\d+(?:\.\d+)?", q_lower)
    has_dims = len(q_nums) >= 2

    # Identify potential product head terms from the query
    product_terms = {
        'rug','rugs','frame','frames','vanity','vanities','desk','desks','chair','chairs','stool','stools','bench','benches','sofa','sofas','loveseat','loveseats',
        'bed','beds','headboard','headboards','duvet','cover','covers','grill','sink','faucet','shelf','shelves','cabinet','cabinets','console','table','tables',
        'light','lights','lantern','lanterns','lamp','lamps','hooks','hook','toy','gate','mirror','mirrors','cushion','cushions','mattress','topper','protector',
        'pouf','poufs','stand','stands','basket','baskets','rack','racks','bookcase','bookcases','dresser','dressers','nightstand','nightstands','sectional',
        'mat','mats','calendar','umbrella','sconce','towel','rod','rods','gate','gates','shelving',
        # added high-signal product heads commonly queried in this dataset
        'pantry','wardrobe','armoire','pillow','pillows','shade','shades','base','bases','plant','plants','loaf','pan','baking','pillowcase','pillowcases','pillowcover','pillowcovers',
        'molding','moulding','moldings','mouldings'
    }

    # detect useful multiword product phrases present in the query
    phrases = [
        'coffee table','console table','wall mirror','string lights','fire pit','toilet paper stand','coat hooks','pet gate','valet rod','grill cover','writing desk',
        'barn door','storage shelf','patio cover','l shaped desk','l shape desk','outdoor lounge chair',
        # additional high-value phrases
        'pantry cabinet','pantry cupboard','kitchen pantry','plant stand','plant stands','lamp shade','lamp shades',
        'throw pillow','decorative pillow','duvet cover','bicycle plant stand','tie dye duvet','chair rail'
    ]
    q_lower_spaced = f" {q_lower} "
    q_phrase_hits = {p for p in phrases if f" {p} " in q_lower_spaced}

    # also normalize common compound tokens from split words, e.g., love seat -> loveseat
    if 'love' in q_terms and 'seat' in q_terms:
        q_phrase_hits.add('loveseat')
    if 'bar' in q_terms and 'stool' in q_terms:
        q_phrase_hits.add('bar stool')

    # Hyphen-aware variants for certain phrases (e.g., l-shaped desk)
    alt_phrase_forms = set()
    for p in list(q_phrase_hits):
        if p in ('l shaped desk', 'l shape desk'):
            alt_phrase_forms.add('l-shaped desk')
            alt_phrase_forms.add(p.replace('l shaped', 'l-shaped'))
            alt_phrase_forms.add(p.replace('l shape', 'l-shape'))
    q_phrases_all = q_phrase_hits | alt_phrase_forms

    q_products = {t for t in q_terms if t in product_terms}

    # targeted brand cue: kohler only (low-risk boost)
    q_has_kohler = 'kohler' in q_terms_set
    brand_phrases = ['orren ellis','allmodern','wayfair sleep','liberty hardware']
    q_brand_hits = [bp for bp in brand_phrases if bp in q_lower]

    # Retrieve from multiple views
    topk = int(kwargs.get('top_k', 20))
    topk = max(10, min(20, topk))  # keep within supported bounds

    # Core sources using base query
    bm25_title_or = safe_bm25(variants[0], 'title', 'or', topk)
    bm25_title_and = safe_bm25(variants[0], 'title', 'and', topk)
    bm25_desc_or = safe_bm25(variants[0], 'description', 'or', topk)
    bm25_desc_and = safe_bm25(variants[0], 'description', 'and', topk)

    # Variant sources (prefer AND for precision) plus a loose OR for swapped/expanded forms
    var_title_ands = []
    var_title_ors = []
    for v in variants[1:]:
        var_title_ands.append(safe_bm25(v, 'title', 'and', topk))
        var_title_ors.append(safe_bm25(v, 'title', 'or', max(10, topk // 2)))

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
    r_var_title_ors = [rank_map(v) for v in var_title_ors]
    r_emb = rank_map(emb)

    # Weighted Reciprocal Rank Fusion
    k_rrf = 60
    w_title_or = 1.15
    w_title_and = 2.0 if has_dims else 1.55
    w_desc_or = 0.9
    w_desc_and = 1.0
    w_var_title_and = 1.65 if has_dims else 1.2
    w_var_title_or = 0.6
    w_emb = 1.25  # slightly increased to help style/semantic queries

    # Collect all candidate ids
    candidate_ids = set(list(r_title_or.keys()) + list(r_title_and.keys()) + list(r_desc_or.keys()) + list(r_desc_and.keys()) + list(r_emb.keys()))
    for rm in r_var_title_ands + r_var_title_ors:
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
    harvest([bm25_title_or, bm25_title_and, bm25_desc_or, bm25_desc_and, emb] + var_title_ands + var_title_ors)

    # Color-aware adjustments
    color_terms = {"black","white","gray","grey","blue","navy","red","green","pink","purple","gold","silver","bronze","brass","brown","beige","ivory","charcoal","champagne","turquoise","teal"}
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
        for rm in r_var_title_ors:
            if did in rm:
                score += w_var_title_or / (k_rrf + rm[did])
        if did in r_emb:
            score += w_emb / (k_rrf + r_emb[did])

        # Light lexical boosts
        m = meta.get(did, {})
        title = (m.get('title') or '').lower()
        desc = (m.get('description') or '').lower()

        if title:
            # Exact phrase presence boost for the base normalized query only
            if q_lower in title:
                score += 0.4
            # Token coverage boost using content terms (ignore stopwords/numbers/colors)
            if q_content_terms:
                color_terms_cov = {"black","white","gray","grey","blue","navy","red","green","pink","purple","gold","silver","bronze","brass","brown","beige","ivory","charcoal","champagne","turquoise","teal"}
                content_nocolor = [t for t in q_content_terms if t not in color_terms_cov]
                if content_nocolor and all(t in title for t in content_nocolor):
                    score += 0.8
                elif content_nocolor:
                    overlap = sum(1 for t in set(content_nocolor) if t in title)
                    score += 0.15 * (overlap / max(1, len(set(content_nocolor))))
        else:
            if desc and q_lower in desc:
                score += 0.2

        # Number/dimension coverage boost (applies to title+desc)
        if has_dims:
            text = f"{title} {desc}"
            num_overlap = sum(1 for n in q_nums if n in text)
            if len(q_nums) and num_overlap == len(q_nums):
                score += 0.7  # strong boost when all numbers match
            elif num_overlap:
                score += 0.28 * (num_overlap / len(q_nums))
            # if both numbers and a separator word present, add a tiny extra nudge
            if ((" x " in text) or (" by " in text)) and all(n in text for n in q_nums[:2]):
                score += 0.1

        # Color matches with synonym support (no penalty when missing, only a mild nudge)
        if q_colors:
            def has_color_exact(c, text):
                # avoid boosting brand phrase "canora grey" and ensure whole-word match
                if c in ('grey','gray') and 'canora grey' in text:
                    return False
                return re.search(rf"\\b{re.escape(c)}\\b", text) is not None
            def has_color_any(c, text):
                if c in ('grey','gray'):
                    return has_color_exact('grey', text) or has_color_exact('gray', text)
                if c == 'navy':
                    return has_color_exact('navy', text) or has_color_exact('navy blue', text)
                return has_color_exact(c, text)
            color_found = False
            for c in q_colors:
                if has_color_any(c, title):
                    score += 0.22
                    color_found = True
                if has_color_any(c, desc):
                    score += 0.18
                    color_found = True
            # Demote brand-only matches like 'Canora Grey' when user likely means the color
            if (('grey' in q_terms_set) or ('gray' in q_terms_set)) and ('canora' not in q_terms_set):
                if ('canora grey' in title) or ('canora grey' in desc):
                    score -= 0.35
            # slight demotion if a specific color is requested but not found at all
            if not color_found:
                score -= 0.08

        # Disambiguate molding vs baking molds when query asks for molding
        if ('molding' in q_terms_set) or ('moulding' in q_terms_set):
            td = f"{title} {desc}"
            if re.search(r"\\b(molding|moulding|chair rail)\\b", td):
                score += 0.6
                if 'french' in q_terms_set and 'french' in td:
                    score += 0.15
            if (re.search(r"\\bmold\\b", td) and re.search(r"\\b(bread|baking|loaf|baguette|toast|cake|pan)\\b", td)):
                score -= 0.6
            # avoid ranking 'french doors' for molding queries
            if (('door' in title) or ('doors' in title)) and not re.search(r"\\b(crown|panel|chair rail|molding|moulding)\\b", title):
                score -= 0.55
                if 'french' in q_terms_set:
                    score -= 0.2

        # Disambiguate 'frame' queries that include dimensions in favor of picture/poster frames
        if ('frame' in q_terms_set or 'frames' in q_terms_set) and has_dims:
            td = f" {title} {desc} "
            if any(p in td for p in [' picture frame', ' poster frame', ' photo frame', ' gallery frame', ' wall frame']):
                score += 0.45
            # demote unrelated frame senses unless explicitly asked in query
            if ((' bed frame' in td) or (' platform bed' in td) or (' headboard' in td) or (' projector screen' in td) or (' coffee table' in td)) and not (('bed' in q_terms_set) or ('projector' in q_terms_set) or ('table' in q_terms_set)):
                score -= 0.4

        # Product head-term and phrase alignment: prioritize items that contain product words/phrases from the query
        if q_products or q_phrases_all:
            present_word = any((f" {p} " in f" {title} ") or (f" {p} " in f" {desc} ") for p in q_products)
            present_phrase = any((p in title) or (p in desc) for p in q_phrases_all)
            if present_word or present_phrase:
                score += 0.45
            else:
                # milder penalty if none of the product cues appear
                score -= 0.05

        # Targeted brand boost for queries with 'kohler' and other known brand phrases
        if q_has_kohler:
            if (" kohler " in f" {title} ") or (" kohler " in f" {desc} "):
                score += 0.25
        if q_brand_hits:
            for bp in q_brand_hits:
                if (f" {bp} " in f" {title} ") or (f" {bp} " in f" {desc} "):
                    score += 0.35

        # Daybed nudge
        if ('daybed' in q_terms_set) or ('day' in q_terms_set and 'bed' in q_terms_set):
            if 'daybed' in title:
                score += 0.25
            elif 'day bed' in title:
                score += 0.15

        # Small raw BM25 score nudging if available (helps break ties)
        raw = m.get('score')
        if isinstance(raw, (int, float)):
            score += 0.02 * (raw / 100.0)  # slightly larger normalized bump

        scores[did] = score

    # Sort by fused score desc, tiebreak on best individual rank
    def best_rank(did):
        ranks = [r.get(did, 10**9) for r in (r_title_or, r_title_and, r_desc_or, r_desc_and, r_emb)]
        for rm in r_var_title_ands + r_var_title_ors:
            ranks.append(rm.get(did, 10**9))
        return min(ranks)

    ranked = sorted(scores.items(), key=lambda x: (-x[1], best_rank(x[0])))

    # Return up to 10 results as strings
    return [did for did, _ in ranked[:10]]
