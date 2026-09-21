from __future__ import annotations

import hashlib
import json


class DummyEnricher:
    """Deterministic enrichment engine used for tests and baselines."""

    def __init__(self, vocabulary: list[str]):
        self.vocabulary = vocabulary

    def enrich(self, query: str) -> list[str]:
        del query
        return self.vocabulary[:1]

    @property
    def cache_key(self) -> str:
        payload = {"type": "dummy", "vocabulary": self.vocabulary}
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()


def make_dummy_enricher(vocabulary: list[str]) -> DummyEnricher:
    return DummyEnricher(vocabulary)
