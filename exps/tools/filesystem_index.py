from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from exps.paths import SEARCH_EXPERIMENTS_ROOT
from exps.tools.filesystem import (
    _default_path_builder,
    _ensure_filesystem_columns,
    _wands_path_builder,
)


def filesystem_root(dataset_name: str) -> Path:
    return SEARCH_EXPERIMENTS_ROOT / "filesystem" / dataset_name


def ensure_filesystem_on_disk(
    corpus,
    *,
    dataset_name: str,
    variant: str = "default",
) -> Path:
    base_dir = filesystem_root(dataset_name)
    if base_dir.exists():
        return base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    if variant == "wands":
        path_builder = _wands_path_builder
    else:
        path_builder = _default_path_builder
    _ensure_filesystem_columns(corpus, variant=variant, path_builder=path_builder)
    paths = corpus["path"].tolist()
    contents_list = corpus["contents"].tolist()
    for rel_path, contents in tqdm(
        zip(paths, contents_list),
        total=len(paths),
        desc=f"Writing filesystem dataset {dataset_name}",
    ):
        rel_path = str(rel_path).lstrip("/")
        file_path = base_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(str(contents), encoding="utf-8")
    return base_dir
