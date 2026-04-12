import pickle
from pathlib import Path

import pandas as pd
from searcharray import SearchArray

from cheat_at_search import esci_data, msmarco_data, wands_data
from cheat_at_search.tokenizers import snowball_tokenizer

DATASETS = {
    "esci": esci_data,
    "msmarco": msmarco_data,
    "wands": wands_data,
}

BM25_PARAMS = {
    "msmarco": {"k1": 0.9, "b": 0.4},
}

DEFAULT_BM25_PARAMS = {"k1": 1.2, "b": 0.75}

SNOWBALL_FIELDS = ("title", "description", "category")
CACHE_ROOT = Path.home() / ".search-experiments" / "searcharray"
BM25_CACHE_ROOT = Path.home() / ".search-experiments" / "bm25"


def _cache_path(dataset_name: str, field: str) -> Path:
    return CACHE_ROOT / dataset_name / f"{field}_snowball.pkl"


def _load_cached_index(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.UnpicklingError):
        return None


def _save_cached_index(path: Path, index: SearchArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(index, handle)


def _ensure_cached_field(corpus, dataset_name: str, field: str, workers: int) -> None:
    snowball_field = f"{field}_snowball"
    if snowball_field in corpus or field not in corpus:
        return

    cache_path = _cache_path(dataset_name, field)
    cached = _load_cached_index(cache_path)
    if cached is None:
        cached = SearchArray.index(corpus[field], snowball_tokenizer, workers=workers)
        _save_cached_index(cache_path, cached)
    corpus[snowball_field] = cached


def _bm25_cache_path(
    dataset_name: str, num_queries: int | None, seed: int | None
) -> Path:
    num_queries_label = num_queries if num_queries is not None else "all"
    seed_label = seed if seed is not None else "none"
    filename = f"graded_bm25_n{num_queries_label}_seed{seed_label}.pkl"
    return BM25_CACHE_ROOT / dataset_name / filename


def load_bm25_cache(
    dataset_name: str, num_queries: int | None, seed: int | None
) -> pd.DataFrame | None:
    cache_path = _bm25_cache_path(dataset_name, num_queries, seed)
    if not cache_path.exists():
        return None
    try:
        graded = pd.read_pickle(cache_path)
    except (OSError, pickle.UnpicklingError):
        return None
    if "doc_id" not in graded.columns:
        return None
    if "mrr" not in graded.columns:
        return None
    return graded


def save_bm25_cache(
    dataset_name: str,
    num_queries: int | None,
    seed: int | None,
    graded: pd.DataFrame,
) -> None:
    cache_path = _bm25_cache_path(dataset_name, num_queries, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    graded.to_pickle(cache_path)


def get_dataset(name: str, workers: int = 1):
    try:
        dataset = DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc

    corpus = dataset.corpus
    for field in SNOWBALL_FIELDS:
        _ensure_cached_field(corpus, name, field, workers)
    return dataset


def bm25_params_for_dataset(name: str) -> tuple[float, float]:
    params = BM25_PARAMS.get(name, DEFAULT_BM25_PARAMS)
    return params["k1"], params["b"]
