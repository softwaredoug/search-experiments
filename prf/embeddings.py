from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


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


def _row_text(row) -> str:
    title = row.get("title")
    description = row.get("description")
    title_text = title.strip() if isinstance(title, str) else ""
    description_text = description.strip() if isinstance(description, str) else ""
    if title_text and description_text:
        return f"{title_text}\n\n{description_text}"
    if title_text:
        return title_text
    return description_text


def _corpus_signature(corpus, model_name: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
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


def _cache_paths(signature: str) -> tuple[Path, Path]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    emb_path = CACHE_ROOT / f"embeddings_{signature}.npy"
    meta_path = CACHE_ROOT / f"embeddings_{signature}.json"
    return emb_path, meta_path


def _manifest_path(signature: str) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"embeddings_{signature}.manifest.json"


def _chunk_path(signature: str, chunk_index: int) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"embeddings_{signature}_chunk_{chunk_index}.npy"


def _load_cache(signature: str, model_name: str):
    emb_path, meta_path = _cache_paths(signature)
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


def _load_manifest(signature: str, model_name: str):
    manifest_path = _manifest_path(signature)
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


def _save_manifest(signature: str, meta: dict) -> None:
    manifest_path = _manifest_path(signature)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle)


def load_or_create_embeddings(
    corpus,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    show_progress: bool = False,
) -> np.ndarray:
    signature = _corpus_signature(corpus, model_name)
    cached = _load_cache(signature, model_name)
    if cached is not None:
        return cached

    total_count = len(corpus)
    manifest = _load_manifest(signature, model_name)
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

    for chunk_index in range(num_chunks):
        if show_progress and num_chunks > 0:
            print(f"Embedding chunks {chunk_index + 1}/{num_chunks}", end="\r", flush=True)
        chunk_file = _chunk_path(signature, chunk_index)
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
        texts = [_row_text(row) for _, row in corpus.iloc[start:end].iterrows()]
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
        _save_manifest(signature, manifest)

    if show_progress and num_chunks > 0:
        print("")
    if embeddings is None:
        embeddings = np.empty((total_count, dim or 0))
    manifest.update({
        "dim": dim,
        "count": total_count,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "completed_chunks": sorted(completed),
    })
    _save_manifest(signature, manifest)
    return embeddings


def cache_paths_for_corpus(
    corpus, model_name: str = DEFAULT_MODEL_NAME
) -> tuple[Path, Path]:
    signature = _corpus_signature(corpus, model_name)
    return _manifest_path(signature), _chunk_path(signature, 0)
