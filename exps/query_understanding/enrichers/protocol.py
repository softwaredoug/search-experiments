from __future__ import annotations

from typing import Protocol


class Enricher(Protocol):
    @property
    def cache_key(self) -> str:
        ...

    def enrich(self, query: str) -> list[str]:
        ...
