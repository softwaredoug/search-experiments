from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pickle


CACHE_ROOT = Path.home() / ".search-experiments" / "cache"


def _cache_key(
    *,
    dataset: str,
    strategy_type: str,
    params: dict,
    num_queries: int | None,
    seed: int | None,
) -> str:
    payload = {
        "dataset": dataset,
        "strategy_type": strategy_type,
        "params": params,
        "num_queries": num_queries,
        "seed": seed,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"strategy_results_{key}.pkl"


def load_cached_results(
    *,
    dataset: str,
    strategy_type: str,
    params: dict,
    num_queries: int | None,
    seed: int | None,
) -> pd.DataFrame | None:
    key = _cache_key(
        dataset=dataset,
        strategy_type=strategy_type,
        params=params,
        num_queries=num_queries,
        seed=seed,
    )
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except (OSError, ValueError, pickle.UnpicklingError):
        return None


def save_cached_results(
    *,
    dataset: str,
    strategy_type: str,
    params: dict,
    num_queries: int | None,
    seed: int | None,
    graded: pd.DataFrame,
) -> None:
    key = _cache_key(
        dataset=dataset,
        strategy_type=strategy_type,
        params=params,
        num_queries=num_queries,
        seed=seed,
    )
    path = _cache_path(key)
    graded.to_pickle(path)
