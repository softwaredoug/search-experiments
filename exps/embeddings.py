from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 10000
CACHE_ROOT = Path.home() / ".search-experiments" / "embeddings"


def _minilm_model(model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding search. "
            "Install it with `uv add sentence-transformers` or `pip install sentence-transformers`."
        ) from exc
    if device:
        return SentenceTransformer(model_name, device=device)
    return SentenceTransformer(model_name)


def _row_text(row, document_prefix: str | None = None) -> str:
    title = row.get("title")
    description = row.get("description")
    title_text = title.strip() if isinstance(title, str) else ""
    description_text = description.strip() if isinstance(description, str) else ""
    if title_text and description_text:
        text = f"{title_text}\n\n{description_text}"
    elif title_text:
        text = title_text
    else:
        text = description_text
    if document_prefix:
        return f"{document_prefix}{text}"
    return text


def _corpus_signature(corpus, model_name: str, document_prefix: str | None = None) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"|")
    if document_prefix:
        hasher.update(document_prefix.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(str(len(corpus)).encode("utf-8"))
    if "doc_id" in corpus.columns:
        for value in corpus["doc_id"].tolist():
            hasher.update(str(value).encode("utf-8"))
            hasher.update(b"|")
    else:
        for value in corpus.index.tolist():
            hasher.update(str(value).encode("utf-8"))
            hasher.update(b"|")
    return hasher.hexdigest()


def _cache_root(dataset_name: str | None) -> Path:
    root = CACHE_ROOT if dataset_name is None else CACHE_ROOT / dataset_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_paths(signature: str, root: Path) -> tuple[Path, Path]:
    emb_path = root / f"embeddings_{signature}.npy"
    meta_path = root / f"embeddings_{signature}.json"
    return emb_path, meta_path


def _manifest_path(signature: str, root: Path) -> Path:
    return root / f"embeddings_{signature}.manifest.json"


def _chunk_path(signature: str, chunk_index: int, root: Path) -> Path:
    return root / f"embeddings_{signature}_chunk_{chunk_index}.npy"


def _load_cache(signature: str, model_name: str, root: Path):
    emb_path, meta_path = _cache_paths(signature, root)
    if not emb_path.exists() or not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        embeddings = np.load(emb_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if meta.get("signature") != signature:
        return None
    if embeddings.shape[0] != meta.get("count"):
        return None
    if embeddings.shape[1] != meta.get("dim"):
        return None
    if meta.get("model") != model_name:
        return None
    return embeddings


def _load_manifest(signature: str, model_name: str, root: Path):
    manifest_path = _manifest_path(signature, root)
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("signature") != signature:
        return None
    if meta.get("model") != model_name:
        return None
    return meta


def _save_manifest(signature: str, meta: dict, root: Path) -> None:
    manifest_path = _manifest_path(signature, root)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle)


def load_or_create_embeddings(
    corpus,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    show_progress: bool = True,
    dataset_name: str | None = None,
    document_prefix: str | None = None,
) -> np.ndarray:
    signature = _corpus_signature(corpus, model_name, document_prefix)
    root = _cache_root(dataset_name)
    cached = _load_cache(signature, model_name, root)
    if cached is not None:
        return cached
    if dataset_name is not None:
        legacy_cached = _load_cache(signature, model_name, _cache_root(None))
        if legacy_cached is not None:
            return legacy_cached

    total_count = len(corpus)
    manifest = _load_manifest(signature, model_name, root)
    if manifest is None:
        manifest = {
            "signature": signature,
            "model": model_name,
            "dim": None,
            "count": total_count,
            "chunk_size": chunk_size,
            "num_chunks": int(math.ceil(total_count / chunk_size)) if chunk_size > 0 else 0,
            "completed_chunks": [],
        }
    else:
        chunk_size = int(manifest.get("chunk_size", chunk_size))

    num_chunks = int(math.ceil(total_count / chunk_size)) if chunk_size > 0 else 0
    dim = manifest.get("dim")
    embeddings = None
    model = None
    completed = set(manifest.get("completed_chunks", []))

    chunk_iter = range(num_chunks)
    if show_progress and num_chunks > 0:
        chunk_iter = tqdm(chunk_iter, desc="Embedding chunks")
    for chunk_index in chunk_iter:
        chunk_file = _chunk_path(signature, chunk_index, root)
        start = chunk_index * chunk_size
        end = min(start + chunk_size, total_count)
        if chunk_file.exists():
            chunk = np.load(chunk_file)
            if dim is None:
                dim = int(chunk.shape[1])
                embeddings = np.empty((total_count, dim), dtype=chunk.dtype)
            if embeddings is None:
                embeddings = np.empty((total_count, dim), dtype=chunk.dtype)
            embeddings[start:end] = chunk
            completed.add(chunk_index)
            continue

        if model is None:
            model = _minilm_model(model_name, device=device)
        texts = [
            _row_text(row, document_prefix=document_prefix)
            for _, row in corpus.iloc[start:end].iterrows()
        ]
        chunk = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        if chunk.ndim != 2:
            chunk = np.asarray(chunk)
        if dim is None:
            dim = int(chunk.shape[1])
            embeddings = np.empty((total_count, dim), dtype=chunk.dtype)
        if embeddings is None:
            embeddings = np.empty((total_count, dim), dtype=chunk.dtype)
        embeddings[start:end] = chunk
        np.save(chunk_file, chunk)
        completed.add(chunk_index)
        manifest.update({
            "dim": dim,
            "count": total_count,
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "completed_chunks": sorted(completed),
        })
        _save_manifest(signature, manifest, root)

    if embeddings is None:
        embeddings = np.empty((total_count, dim or 0))
    manifest.update({
        "dim": dim,
        "count": total_count,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "completed_chunks": sorted(completed),
    })
    _save_manifest(signature, manifest, root)
    return embeddings


def cache_paths_for_corpus(
    corpus,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_name: str | None = None,
    document_prefix: str | None = None,
) -> tuple[Path, Path]:
    signature = _corpus_signature(corpus, model_name, document_prefix)
    root = _cache_root(dataset_name)
    return _manifest_path(signature, root), _chunk_path(signature, 0, root)
