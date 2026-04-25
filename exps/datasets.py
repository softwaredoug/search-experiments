import pickle
from pathlib import Path
from searcharray import SearchArray
from typing_extensions import Literal

from cheat_at_search.tokenizers import snowball_tokenizer

from exps.mounting import ensure_data_mounted

SNOWBALL_FIELDS = ("title", "description", "category")
DATASET_NAMES = ("esci", "minimarco", "msmarco", "wands")
DatasetName = Literal["esci", "minimarco", "msmarco", "wands"]
CACHE_ROOT = Path.home() / ".search-experiments" / "searcharray"

def _cache_path(dataset_name: str, field: str) -> Path:
    return CACHE_ROOT / dataset_name / f"{field}_snowball.pkl"


def _load_cached_index(path: Path, expected_length: int | None = None):
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            cached = pickle.load(handle)
    except (OSError, pickle.UnpicklingError):
        return None
    if expected_length is not None and len(cached) != expected_length:
        return None
    return cached


def _save_cached_index(path: Path, index: SearchArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(index, handle)


def _ensure_cached_field(corpus, dataset_name: str, field: str, workers: int) -> None:
    snowball_field = f"{field}_snowball"
    if snowball_field in corpus or field not in corpus:
        return

    cache_path = _cache_path(dataset_name, field)
    cached = _load_cached_index(cache_path, expected_length=len(corpus))
    if cached is None:
        cached = SearchArray.index(corpus[field], snowball_tokenizer, workers=workers)
        _save_cached_index(cache_path, cached)
    corpus[snowball_field] = cached




def get_dataset(name: DatasetName, workers: int = 1, ensure_snowball: bool = True):
    ensure_data_mounted()
    try:
        if name == "esci":
            from cheat_at_search import esci_data as dataset
        elif name == "msmarco":
            from cheat_at_search import msmarco_data as dataset
        elif name == "minimarco":
            from cheat_at_search import minimarco_data as dataset
        elif name == "wands":
            from cheat_at_search import wands_data as dataset
        else:
            raise KeyError(name)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc

    corpus = dataset.corpus
    if ensure_snowball:
        for field in SNOWBALL_FIELDS:
            _ensure_cached_field(corpus, name, field, workers)
    return dataset
