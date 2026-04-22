from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_ROOT = Path.home() / ".search-experiments" / "embeddings"


def _minilm_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding search. "
            "Install it with `uv add sentence-transformers` or `pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer(DEFAULT_MODEL_NAME)


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


def _corpus_signature(corpus) -> str:
    hasher = hashlib.sha256()
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


def _load_cache(signature: str):
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
    if meta.get("model") != DEFAULT_MODEL_NAME:
        return None
    return embeddings


def _save_cache(signature: str, embeddings: np.ndarray) -> None:
    emb_path, meta_path = _cache_paths(signature)
    meta = {
        "signature": signature,
        "model": DEFAULT_MODEL_NAME,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
    }
    np.save(emb_path, embeddings)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle)


def load_or_create_embeddings(corpus) -> np.ndarray:
    signature = _corpus_signature(corpus)
    cached = _load_cache(signature)
    if cached is not None:
        return cached

    model = _minilm_model()
    texts = [_row_text(row) for _, row in corpus.iterrows()]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    if embeddings.ndim != 2:
        embeddings = np.asarray(embeddings)
    _save_cache(signature, embeddings)
    return embeddings
