from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pickle


CACHE_ROOT = Path.home() / ".search-experiments" / "cache"
CHUNK_SIZE = 100


def cache_key(
    *,
    dataset: str,
    strategy_type: str,
    params: dict,
    seed: int | None,
    query_list_hash: str | None = None,
) -> str:
    payload = {
        "dataset": dataset,
        "strategy_type": strategy_type,
        "params": params,
        "seed": seed,
        "query_list_hash": query_list_hash,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"strategy_results_{key}.pkl"


def manifest_path(key: str) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"strategy_results_{key}.manifest.json"


def chunk_path(key: str, chunk_index: int) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"strategy_results_{key}_chunk_{chunk_index}.pkl"


def load_manifest(key: str) -> dict | None:
    path = manifest_path(key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def save_manifest(key: str, manifest: dict) -> None:
    path = manifest_path(key)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def load_chunk(key: str, chunk_index: int) -> pd.DataFrame | None:
    path = chunk_path(key, chunk_index)
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except (OSError, ValueError, pickle.UnpicklingError):
        return None


def save_chunk(key: str, chunk_index: int, graded: pd.DataFrame) -> None:
    path = chunk_path(key, chunk_index)
    graded.to_pickle(path)


def load_cached_results(
    *,
    dataset: str,
    strategy_type: str,
    params: dict,
    num_queries: int | None,
    seed: int | None,
    query_list_hash: str | None = None,
) -> pd.DataFrame | None:
    key = cache_key(
        dataset=dataset,
        strategy_type=strategy_type,
        params=params,
        seed=seed,
        query_list_hash=query_list_hash,
    )
    manifest = load_manifest(key)
    if manifest is not None:
        total_chunks = int(manifest.get("num_chunks", 0))
        completed = set(manifest.get("completed_chunks", []))
        if total_chunks and len(completed) == total_chunks:
            chunks = []
            for idx in range(total_chunks):
                chunk = load_chunk(key, idx)
                if chunk is None:
                    return None
                chunks.append(chunk)
            if not chunks:
                return None
            return pd.concat(chunks, ignore_index=True)

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
    query_list_hash: str | None = None,
    graded: pd.DataFrame,
) -> None:
    key = cache_key(
        dataset=dataset,
        strategy_type=strategy_type,
        params=params,
        seed=seed,
        query_list_hash=query_list_hash,
    )
    path = _cache_path(key)
    graded.to_pickle(path)
