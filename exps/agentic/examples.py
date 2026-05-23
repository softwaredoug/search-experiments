from __future__ import annotations

import random


def _grade_column(judgments):
    for col in ("grade", "relevance", "rel", "label", "score"):
        if col in judgments.columns:
            return col
    return None


def _grade_to_emoji(grade, grade_levels):
    if not grade_levels:
        return "😐"
    if len(grade_levels) == 1:
        return "😐"
    if grade == grade_levels[0]:
        return "😭"
    if grade == grade_levels[-1]:
        return "😃"
    return "😐"


def _sorted_grades(values):
    def _coerce(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    numeric = [value for value in values if _coerce(value) is not None]
    if len(numeric) == len(values):
        return sorted(numeric)
    return sorted(values, key=lambda value: str(value))


def append_few_shot_examples(
    system_prompt: str,
    *,
    corpus,
    judgments,
    few_shot_config: list,
) -> str:
    if judgments is None:
        raise ValueError("few_shot requires judgments to be available.")
    grade_col = _grade_column(judgments)
    if grade_col is None:
        raise ValueError("few_shot requires a grade column in judgments.")
    if "query" not in judgments.columns or "doc_id" not in judgments.columns:
        raise ValueError("few_shot requires query and doc_id columns in judgments.")

    corpus_lookup = None
    if "doc_id" in corpus.columns:
        corpus_lookup = corpus.set_index("doc_id", drop=False)

    blocks = []
    for entry in few_shot_config:
        if isinstance(entry, dict) and "sample_judgments" in entry:
            raw = entry["sample_judgments"]
            if isinstance(raw, dict):
                if "num_rows" not in raw:
                    raise ValueError("few_shot.sample_judgments requires num_rows.")
                sample_count = int(raw["num_rows"])
                columns = raw.get("columns") or []
            else:
                sample_count = int(raw)
                columns = []
            if sample_count <= 0:
                continue
            if not isinstance(columns, list):
                raise ValueError("few_shot.sample_judgments.columns must be a list.")
            for col in columns:
                if col not in corpus.columns:
                    raise ValueError(f"few_shot column not found in corpus: {col}")
            seed = int(entry.get("seed", 42))

            pool = judgments.dropna(subset=[grade_col, "query", "doc_id"])
            grades = list(pool[grade_col].dropna().unique())
            grades = _sorted_grades(grades)
            if not grades:
                continue
            rng = random.Random(seed)
            grouped = {
                grade: pool[pool[grade_col] == grade].sample(
                    frac=1.0, random_state=rng.randrange(1 << 30)
                )
                for grade in grades
            }
            queues = {grade: grouped[grade].iterrows() for grade in grades}
            samples = []
            while len(samples) < sample_count:
                advanced = False
                for grade in grades:
                    try:
                        _, row = next(queues[grade])
                    except StopIteration:
                        continue
                    samples.append(row)
                    advanced = True
                    if len(samples) >= sample_count:
                        break
                if not advanced:
                    break

            lines = [
                "Few-shot examples (query, product, relevance):",
            ]
            for row in samples:
                query = row.get("query")
                doc_id = row.get("doc_id")
                grade = row.get(grade_col)
                emoji = _grade_to_emoji(grade, grades)
                title = ""
                description = ""
                extra_fields = {}
                if corpus_lookup is not None and doc_id in corpus_lookup.index:
                    match = corpus_lookup.loc[doc_id]
                    if hasattr(match, "ndim") and match.ndim > 1:
                        match = match.iloc[0]
                    if hasattr(match, "get"):
                        title = match.get("title", "")
                        description = match.get("description", "")
                        for col in columns:
                            extra_fields[col] = match.get(col, "")
                lines.extend(
                    [
                        f"Query: {query}",
                        f"Doc ID: {doc_id}",
                        f"Title: {title}",
                        f"Description: {description}",
                    ]
                )
                for col in columns:
                    lines.append(f"{col}: {extra_fields.get(col, '')}")
                lines.extend(
                    [
                        f"Relevance: {emoji}",
                        "",
                    ]
                )
            blocks.append("\n".join(lines).strip())
            continue
        raise ValueError("few_shot entries must be mappings with sample_judgments.")

    if not blocks:
        return system_prompt
    return system_prompt.rstrip() + "\n\n" + "\n\n".join(blocks) + "\n"
