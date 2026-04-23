from __future__ import annotations

from typing import Iterable


def build_doc_id_lookup(corpus) -> dict[str, int]:
    if "doc_id" not in corpus.columns:
        return {}
    return {str(doc_id): int(index) for index, doc_id in corpus["doc_id"].items()}


def doc_ids_to_indices(doc_ids: Iterable[int | str], lookup: dict[str, int]) -> list[int]:
    indices: list[int] = []
    for doc_id in doc_ids:
        index = lookup.get(str(doc_id))
        if index is not None:
            indices.append(index)
    return indices
