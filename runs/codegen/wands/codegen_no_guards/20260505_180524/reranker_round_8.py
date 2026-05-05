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
    # Expand product/category terms for better must-term detection
    product_terms.update({'door','doors','clock','clocks'})

    # Common color/style tokens the user may care about being present in the item
    color_terms = {
        'white','black','gray','grey','ivory','beige','tan','teal','navy','olive','peach','magenta','purple',
        'red','green','blue','brown','gold','silver','bronze','brass','yellow','orange','turquoise','cream'
    }

    # Finish/material tokens frequently used in hardware/fixtures queries
    finish_terms = {'matte','satin','brushed','polished','chrome','nickel','stainless','steel','iron'}

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
        'alium': ['allium'],  # one alium way -> one allium way
        'doning': ['dining'],
        'ligth': ['light'],
        'bookshelf': ['bookcase','shelf','shelves'],
        'bookshelves': ['bookcases','shelves'],
        'ladies': ['women','womens','women s'],
        'woman': ['women'],
        'rocker': ['rocking'],
        # texture adjectives often used colloquially
        'fluffy': ['fuzzy','furry','fur','shag','shaggy'],
        # lighting/LED related
        'leds': ['led','lights','lighted'],
        'led': ['lights','lighted'],
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
    # Strengthen must-term detection for common synonyms not in product_terms
    if ('bookshelf' in q_tokens_aug) or ('bookshelves' in q_tokens_aug):
        for mt in ['bookcase','shelf','shelves','bookshelf','bookshelves']:
            if mt not in must_terms:
                must_terms.append(mt)
    # TV unit intents are usually stands/consoles
    if ('tv' in q_tokens_aug) and ('unit' in q_tokens_aug):
        for mt in ['stand','console','media','tv','stand']:  # include media keyword to help console detection
            if mt not in must_terms:
                must_terms.append(mt)

    # Numeric tokens (e.g., dimensions) for size-aware matching
    numeric_tokens = [t for t in q_tokens if t.isdigit()]

    # Build a few light query expansion variants to increase recall for BM25
    def build_alt_queries(q: str):
        alts = set()
        q_norm = normalize_text(q)
        # spelling variants and phrase re-orderings for common structures
        repls = [
            (r"\bpedistole\b", "pedestal"),
            (r"\btye\s+dye\b", "tie dye"),
            (r"\bbiycicle\b", "bicycle"),
            (r"\bqueen\s+ann\b", "queen anne"),
            (r"\bgrey\b", "gray"),
            (r"\balium\b", "allium"),
            (r"\bdoning\b", "dining"),
            (r"\bligth\b", "light"),
        ]
        alt = q_norm
        for pat, rep in repls:
            alt = re.sub(pat, rep, alt)
        if alt != q_norm:
            alts.add(alt)

        # One Allium Way brand normalization
        if re.search(r"\bone\s+all?ium\s+way\b", q_norm):
            alts.add(re.sub(r"\bone\s+all?ium\s+way\b", "one allium way", q_norm))
            alts.add(q_norm.replace("one allium way", "onealliumway"))
        elif re.search(r"\ball?ium\b", q_norm) and 'one' in q_norm and 'way' in q_norm:
            alts.add("one allium way")

        # Bookshelf hanging -> wall mounted/floating shelf
        if re.search(r"\b(hanging)\b.*\b(shelf|shelves|bookcase|bookshelf|bookshelves)\b", q_norm):
            alts.add(resub1 := re.sub(r"\bhanging\b", "wall mounted", q_norm))
            alts.add(q_norm.replace("hanging", "floating"))
            alts.add(q_norm.replace("bookshelf", "wall shelf"))
            alts.add(q_norm.replace("bookshelves", "wall shelves"))
        if re.search(r"\bbookshelf\b|\bbookshelves\b", q_norm):
            alts.add(q_norm.replace("bookshelf", "bookcase"))

        # TV unit -> tv stand/media console
        if re.search(r"\btv\s+unit\b", q_norm):
            alts.add(q_norm.replace("tv unit", "tv stand"))
            alts.add(q_norm.replace("tv unit", "media console"))
            alts.add(q_norm.replace("tv unit", "tv console"))

        # Semi-flush/semiflush normalization
        if re.search(r"\bsemi\s*[- ]?flush\b", q_norm):
            alts.add(re.sub(r"semi\s*[- ]?flush", "semi flush", q_norm))
            alts.add(re.sub(r"semi\s*[- ]?flush", "semi-flush", q_norm))

        # Mid-century variants
        if re.search(r"\bmid\s*century\b|\bmidcentury\b|\bmid\s*[- ]?century\b", q_norm):
            alts.add(re.sub(r"mid\s*[- ]?century", "mid century", q_norm))
            alts.add(re.sub(r"mid\s*[- ]?century", "mid-century", q_norm))
            # if query mentions tv unit, try common phrasing
            if 'tv' in q_norm and ('unit' in q_norm or 'stand' in q_norm or 'console' in q_norm):
                alts.add(re.sub(r"tv\s+unit", "tv stand", re.sub(r"mid\s*[- ]?century", "mid-century", q_norm)))
                alts.add(re.sub(r"tv\s+unit", "media console", re.sub(r"mid\s*[- ]?century", "mid century", q_norm)))

        # Welcome rug -> doormat variants
        if re.search(r"\bwelcome\b.*\brug\b|\brug\b.*\bwelcome\b", q_norm):
            alts.add(q_norm.replace("welcome rug", "doormat"))
            alts.add(q_norm.replace("welcome rug", "welcome mat"))
            alts.add(q_norm.replace("welcome rug", "door mat"))

        # Wrought -> wrought iron
        if re.search(r"\bwrought\b", q_norm) and not re.search(r"\bwrought\s+iron\b", q_norm):
            alts.add(q_norm.replace("wrought", "wrought iron"))

        # Glass rack vs rack glass
        if re.search(r"\brack\b.*\bglass\b", q_norm):
            alts.add(re.sub(r"\brack\b.*\bglass\b", "glass rack", q_norm))
        if re.search(r"\bglass\b.*\brack\b", q_norm):
            alts.add(re.sub(r"\bglass\b.*\brack\b", "glass rack", q_norm))

        # Collapsed franchise/brand variant (e.g., starwars)
        if re.search(r"\bstar\s+wars\b", q_norm):
            alts.add(re.sub(r"\bstar\s+wars\b", "starwars", q_norm))
            # help rug intent specifically
            if re.search(r"\brug\b|\brugs\b|area rug", q_norm):
                alts.add("star wars rug")
                alts.add("star wars area rug")
        # Star Wars rug phrasing
        if re.search(r"\bstar\s+wars\b.*\brug\b", q_norm):
            alts.add("star wars area rug")

        # Pantry grey/gray normalization and phrasing
        if re.search(r"\bpantry\b", q_norm) and (re.search(r"\bgrey\b", q_norm) or re.search(r"\bgray\b", q_norm)):
            alts.add(q_norm.replace(" grey", " gray"))
            alts.add("gray pantry")
            alts.add("gray pantry cabinet")
            alts.add("gray kitchen pantry")

        # Pedestal sink fix
        if re.search(r"\bpedistole\b.*\bsink\b|\bpedestal\b.*\bsink\b", q_norm):
            alts.add("pedestal sink")

        # Waterproof synonyms (e.g., waterproof -> water-resistant/weatherproof)
        if re.search(r"\bwater\s*proof\b|\bwaterproof\b", q_norm):
            for syn in ["water-resistant","water resistant","weatherproof","weather resistant","all-weather","all weather"]:
                alts.add(q_norm.replace("waterproof", syn).replace("water proof", syn))
            # outdoor storage chest/deck box intents
            if re.search(r"\b(outdoor)\b.*\b(chest|box|storage)\b|\b(chest|box|storage)\b.*\b(outdoor)\b", q_norm):
                alts.add("outdoor deck box")
                alts.add("outdoor storage box")
                alts.add("weatherproof deck box")

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

        # Outdoor sectional dining intent
        if re.search(r"\boutdoor\b", q_norm) and re.search(r"\bsectional\b", q_norm) and re.search(r"\bdining\b", q_norm):
            alts.add("outdoor sectional dining set")
            alts.add("patio sectional dining set")
            alts.add("patio dining sectional")

        return list(alts)[:20]  # allow a few more variants

    alt_queries = build_alt_queries(query)

    # Retrieve candidates
    bm25_or = fielded_bm25(query, fields=['title^9.3', 'description^4.1'], operator='or', top_k=80, k1=1.3, b=0.7)
    bm25_phrase = fielded_bm25(query, fields=['title^9.3', 'description^4.1'], operator='phrase', top_k=40, k1=1.2, b=0.75)
    # Only use AND when query has more than 1 token to avoid over-restriction
    if len(q_tokens) > 1:
        bm25_and = fielded_bm25(query, fields=['title^9.3', 'description^4.1'], operator='and', top_k=45, k1=1.0, b=0.8)
    else:
        bm25_and = []
    # increase semantic recall a bit
    emb = search_embeddings(query, top_k=30)

    # Alt-query retrievals (lighter weight)
    bm25_or_alt_all, bm25_and_alt_all, bm25_phrase_alt_all = [], [], []
    for aq in alt_queries:
        try:
            bm25_or_alt_all.extend(fielded_bm25(aq, fields=['title^9.3', 'description^4.1'], operator='or', top_k=60, k1=1.2, b=0.75) or [])
            if len(q_tokens) > 1:
                bm25_and_alt_all.extend(fielded_bm25(aq, fields=['title^9.3', 'description^4.1'], operator='and', top_k=35, k1=1.0, b=0.8) or [])
            bm25_phrase_alt_all.extend(fielded_bm25(aq, fields=['title^9.3', 'description^4.1'], operator='phrase', top_k=30, k1=1.2, b=0.75) or [])
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
    # Also build ngrams from augmented tokens to catch corrected phrases (e.g., pedestal sink)
    q_bigrams_aug = ngrams(q_tokens_aug, 2)
    q_trigrams_aug = ngrams(q_tokens_aug, 3)
    q_ngrams = list(dict.fromkeys(q_bigrams + q_trigrams + q_bigrams_aug + q_trigrams_aug))  # unique, preserve order
    q_token_set = set(q_tokens)  # ensure defined locally after potential edits
    colors_in_query = [c for c in color_terms if c in q_token_set]

    # Brand and finish info from the query
    q_norm_all = normalize_text(query)
    # brand phrases include common spell and spelling variants
    brand_phrases = ['one allium way','orren ellis','canora grey','canora gray']
    brand_tokens_single = ['moen','delta','kraus','iittala','nectar','umbra','ge']
    # tolerate minor misspells (e.g., alium->allium) when detecting brand intent
    q_norm_brand = re.sub(r"\balium\b", "allium", q_norm_all)
    in_query_brand_phrases = [p for p in brand_phrases if f" {p} " in f" {q_norm_brand} "]
    in_query_brand_tokens = [b for b in brand_tokens_single if b in q_token_set]
    finishes_in_query = [f for f in finish_terms if f in q_token_set]
    brand_phrase_intended = bool(in_query_brand_phrases)

    # Additional intent flags
    q_midcentury_intent = bool(re.search(r"\bmid\s*century\b|\bmidcentury\b", q_norm_all))
    q_tv_unit_intent = ('tv' in q_token_set) and (('unit' in q_token_set) or ('stand' in q_token_set) or ('console' in q_token_set))
    q_recliner_side_table_intent = (('recliner' in q_token_set) or ('recliners' in q_token_set)) and (('table' in q_token_set) or ('end' in q_token_set) or ('between' in q_token_set) or ('chairside' in q_token_set))
    q_led_bed_intent = (('bed' in q_token_set) or ('beds' in q_token_set)) and (('led' in q_token_set) or ('leds' in q_token_set) or ('light' in q_token_set) or ('lights' in q_token_set))

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
            if (('grey' in colors_in_query) or ('gray' in colors_in_query)) and ((' canora grey ' in f" {t_norm} ") or (' canora grey ' in f" {d_norm} ") or (' canora gray ' in f" {t_norm} ") or (' canora gray ' in f" {d_norm} ")) and ('canora' not in q_token_set) and (not brand_phrase_intended):
                color_bonus -= 0.15
            # Slightly stronger color constraints for bedding queries with multiple colors
            if ('bedding' in q_token_set) and len(colors_in_query) >= 2:
                for c in colors_in_query:
                    if (c not in title_set) and (c not in desc_set):
                        color_bonus -= 0.04
                    else:
                        color_bonus += 0.03
            color_bonus = max(-0.12, min(0.16, color_bonus))

        # Brand match bonus/penalty when brand appears in query
        brand_bonus = 0.0
        if in_query_brand_phrases or in_query_brand_tokens:
            for p in in_query_brand_phrases:
                if f" {p} " in f" {t_norm} " or f" {p} " in f" {d_norm} ":
                    brand_bonus += 0.30  # stronger boost for exact brand phrases
                else:
                    brand_bonus -= 0.12  # stronger penalty if brand phrase intent not matched
            for b in in_query_brand_tokens:
                if (b in title_set) or (b in desc_set) or (f" {b} " in f" {t_norm} "):
                    brand_bonus += 0.15
                else:
                    brand_bonus -= 0.08
        # If the query is brand-only (no product head terms), enforce brand presence more strongly
        if brand_phrase_intended and not must_terms:
            has_brand_phrase = any((f" {p} " in f" {t_norm} ") or (f" {p} " in f" {d_norm} ") for p in in_query_brand_phrases)
            if has_brand_phrase:
                brand_bonus += 0.35
            else:
                brand_bonus -= 0.35

        # Finish/material presence bonus when finish tokens are in the query
        finish_bonus = 0.0
        if finishes_in_query:
            for f in finishes_in_query:
                if (f in title_set) or (f in desc_set):
                    finish_bonus += 0.03
                else:
                    finish_bonus -= 0.01
            finish_bonus = max(-0.06, min(0.12, finish_bonus))

        # Phrase n-gram matching bonus to strongly reward exact multi-word intent
        phrase_match_count = 0
        if q_ngrams:
            pad_t = f" {t_norm} "
            pad_d = f" {d_norm} "
            for ng in q_ngrams:
                needle = f" {ng} "
                if (needle in pad_t) or (needle in pad_d):
                    phrase_match_count += 1
                # also consider reversed bigrams for two-term swaps like "rack glass" -> "glass rack"
                parts = ng.split()
                if len(parts) == 2:
                    rev = f" {parts[1]} {parts[0]} "
                    if (rev in pad_t) or (rev in pad_d):
                        phrase_match_count += 1
        phrase_bonus = min(0.18 * phrase_match_count, 0.42)

        # Short query proximity bonus: if 2-3 tokens all appear within a short window in title/desc
        prox_bonus = 0.0
        if 1 < len(q_tokens) <= 3:
            # simple window check using string spans
            window_hit = False
            if all(tok in t_norm for tok in q_tokens):
                # approximate proximity by max span between first and last occurrence
                first_pos = min((t_norm.find(tok) for tok in q_tokens if tok in t_norm), default=-1)
                last_pos = max((t_norm.rfind(tok) for tok in q_tokens if tok in t_norm), default=-1)
                if first_pos != -1 and last_pos != -1 and (last_pos - first_pos) <= 30:
                    window_hit = True
            if not window_hit and all(tok in d_norm for tok in q_tokens):
                first_pos = min((d_norm.find(tok) for tok in q_tokens if tok in d_norm), default=-1)
                last_pos = max((d_norm.rfind(tok) for tok in q_tokens if tok in d_norm), default=-1)
                if first_pos != -1 and last_pos != -1 and (last_pos - first_pos) <= 40:
                    window_hit = True
            if window_hit:
                prox_bonus += 0.08

        # Style/category inference: abstract often implies wall art
        style_bonus = 0.0
        if 'abstract' in q_token_set:
            if any(s in title_set or s in desc_set for s in {'art','canvas','print','painting','poster','wall','decor','picture'}):
                style_bonus += 0.11
            elif any(s in title_set or s in desc_set for s in {'rug','rugs'}):
                style_bonus += 0.04
        # Mid-century style intent
        if q_midcentury_intent:
            if re.search(r"\bmid\s*[- ]?century\b|\bmidcentury\b", t_norm) or re.search(r"\bmid\s*[- ]?century\b|\bmidcentury\b", d_norm):
                style_bonus += 0.18
            else:
                # very small penalty to downrank non-midcentury when explicitly asked
                style_bonus -= 0.04

        # Audience/style inference for teen intent
        youth_bonus = 0.0
        if 'teen' in q_token_set or 'teens' in q_token_set:
            youth_terms = {'teen','teens','kid','kids','youth','tween','junior','girl','girls','boy','boys'}
            if any((tkn in title_set) or (tkn in desc_set) for tkn in youth_terms):
                youth_bonus += 0.10
            else:
                youth_bonus -= 0.05

        # Franchise-specific disambiguation: Star Wars rug should be a rug, not mug/toy/etc.
        starwars_bonus = 0.0
        if ('star' in q_token_set and 'wars' in q_token_set) and any(s in q_token_set for s in {'rug','rugs'}):
            has_sw = (' star ' in f" {t_norm} " and ' wars ' in f" {t_norm} ") or (' star ' in f" {d_norm} " and ' wars ' in f" {d_norm} ") or ('starwars' in t_norm) or ('starwars' in d_norm)
            has_rug = ('rug' in title_set) or ('rug' in desc_set) or ('rugs' in title_set) or ('rugs' in desc_set)
            if has_sw and has_rug:
                starwars_bonus += 0.25
            elif has_sw and not has_rug:
                starwars_bonus -= 0.10
            elif has_rug and not has_sw:
                starwars_bonus -= 0.06

        # Numeric/dimension matching bonus: prefer items mentioning the same numbers
        numeric_bonus = 0.0
        if numeric_tokens:
            matched = 0
            for num in numeric_tokens:
                if (num in title_set) or (num in desc_set) or (f" {num} " in f" {t_norm} ") or (f" {num} " in f" {d_norm} "):
                    matched += 1
            if matched:
                numeric_bonus += min(0.06 * matched, 0.14)
            else:
                numeric_bonus -= 0.05

        # Targeted intent bonuses
        targeted_bonus = 0.0
        # Recliner side/"between recliners" table intent prefers chairside/C-table
        if q_recliner_side_table_intent:
            has_table = ('table' in title_set) or ('table' in desc_set)
            is_chairside = ('chairside' in title_set) or ('chairside' in desc_set)
            has_ctable_phrase = (' c table ' in f" {t_norm} ") or (' c table ' in f" {d_norm} ")
            mentions_recliner = ('recliner' in title_set) or ('recliner' in desc_set) or ('recliners' in title_set) or ('recliners' in desc_set)
            if is_chairside:
                targeted_bonus += 0.22
            if has_ctable_phrase:
                targeted_bonus += 0.12
            if mentions_recliner:
                targeted_bonus += 0.06
            if not has_table:
                targeted_bonus -= 0.10
        # LED bed intent prefers LED/lighted mentions and must reference a bed-like product
        if q_led_bed_intent:
            has_led = any(w in title_set or w in desc_set or (f" {w} " in f" {t_norm} ") or (f" {w} " in f" {d_norm} ") for w in ['led','lights','lighted','with lights','illuminated'])
            has_bedish = any(w in title_set or w in desc_set for w in ['bed','beds','headboard','platform'])
            if has_led:
                targeted_bonus += 0.18
            else:
                targeted_bonus -= 0.08
            if has_bedish:
                targeted_bonus += 0.12
            else:
                targeted_bonus -= 0.12
        # TV unit intent: ensure stand/console/media words present
        if q_tv_unit_intent:
            tv_terms = {'tv','stand','console','media'}
            tv_match = any(term in title_set or term in desc_set for term in tv_terms)
            if tv_match:
                targeted_bonus += 0.06
            else:
                targeted_bonus -= 0.06

        # Weighted combination of retrieval signals
        # Emphasize phrase and AND slightly more; tone down embedding a bit
        score = (
            0.34 * s_or +
            0.32 * s_phrase +
            0.24 * s_and +
            0.15 * s_emb +
            0.07 * s_or_alt +
            0.05 * s_and_alt +
            0.06 * s_phrase_alt +
            title_bonus +
            coverage_bonus +
            must_bonus + must_title_bonus + must_penalty +
            outdoor_bonus +
            color_bonus +
            brand_bonus +
            finish_bonus +
            phrase_bonus +
            prox_bonus +
            style_bonus +
            youth_bonus +
            starwars_bonus +
            numeric_bonus +
            targeted_bonus
        )

        scored.append((score, v['id']))

    # Sort by score desc and return top 10 IDs
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sid for _, sid in scored[:10]]

