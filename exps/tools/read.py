from __future__ import annotations

from typing import Annotated, Any

from exps.tools.results import format_corpus_result


def make_read_tool(corpus):
    """Build a tool that reads one corpus document by its stable ID."""
    doc_ids = corpus["doc_id"] if "doc_id" in corpus.columns else corpus.index
    lookup: dict[str, int] = {}
    for position, doc_id in enumerate(doc_ids):
        key = str(doc_id)
        if key in lookup:
            raise ValueError(f"read tool requires unique document IDs; duplicate: {key}")
        lookup[key] = position

    def read(
        id: Annotated[str, "Stable document ID returned by a search tool."],
    ) -> list[dict[str, Any]]:
        """Read a corpus document by its stable ID."""
        position = lookup.get(str(id))
        if position is None:
            return []
        row = corpus.iloc[position]
        return [format_corpus_result(row, fallback_id=row.name)]

    return read
