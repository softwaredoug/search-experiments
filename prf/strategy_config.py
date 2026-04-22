from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from prf.strategies.agentic import AgenticSearchStrategy
from prf.strategies.bm25 import BM25Strategy


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    type: str
    params: dict[str, Any]


STRATEGY_TYPES = {
    BM25Strategy._type: BM25Strategy,
    AgenticSearchStrategy._type: AgenticSearchStrategy,
}


def load_strategy_config(path: str | Path) -> StrategyConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Strategy config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict) or "strategy" not in raw:
        raise ValueError("Strategy config must contain a 'strategy' mapping.")
    strategy = raw["strategy"]
    if not isinstance(strategy, dict):
        raise ValueError("'strategy' must be a mapping.")
    name = strategy.get("name")
    type_name = strategy.get("type")
    if not name or not type_name:
        raise ValueError(
            "Strategy config requires 'strategy.name' and 'strategy.type'."
        )
    params = strategy.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("'strategy.params' must be a mapping when provided.")
    return StrategyConfig(name=name, type=str(type_name), params=params)


def resolve_strategy_class(type_name: str):
    try:
        return STRATEGY_TYPES[type_name]
    except KeyError as exc:
        raise ValueError(
            "Unknown strategy type. Supported types: "
            + ", ".join(sorted(STRATEGY_TYPES))
        ) from exc
