from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from cheat_at_search.data_dir import DATA_PATH

from exps.query_understanding.enrichers.jev_choice_multiple import (
    JevChoiceMultipleEnricher,
)


class CachedJevChoiceMultipleEnricher:
    """Persist Jev multiple-choice predictions under the shared data directory."""

    def __init__(
        self,
        *,
        field: str,
        vocabulary: list[str],
        choices: dict[str, str | None],
        prompt: str,
        model: str,
        threshold: float,
        reasoning: str | None,
        pad_missing_choices: bool,
    ):
        self.enricher = JevChoiceMultipleEnricher(
            field=field,
            vocabulary=vocabulary,
            choices=choices,
            prompt=prompt,
            model=model,
            threshold=threshold,
            reasoning=reasoning,
            pad_missing_choices=pad_missing_choices,
        )
        self.cache_key = self.enricher.cache_key
        self.cache_path = (
            Path(DATA_PATH)
            / "query_understanding_cache"
            / f"{self.cache_key}.json"
        )
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, list[str]]:
        try:
            contents = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}

        if not isinstance(contents, dict):
            return {}
        return {
            query: categories
            for query, categories in contents.items()
            if isinstance(query, str)
            and isinstance(categories, list)
            and all(isinstance(category, str) for category in categories)
        }

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.cache_path.parent,
                prefix=f".{self.cache_key}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(self._cache, temporary_file, ensure_ascii=False, sort_keys=True)
                temporary_file.write("\n")
            os.replace(temporary_path, self.cache_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def enrich(self, query: str) -> list[str]:
        if query in self._cache:
            return list(self._cache[query])

        categories = self.enricher.enrich(query)
        self._cache[query] = list(categories)
        self._save_cache()
        return list(categories)
