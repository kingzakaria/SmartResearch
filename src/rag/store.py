"""Loading the RAG files: passages (chunks.json) and embeddings (chunk_embeddings.npz)."""
import json

import numpy as np

from .gemini import GeminiClient, load_api_key


def load_chunks(config) -> list[dict]:
    data = json.loads(config.CHUNKS_JSON.read_text(encoding="utf-8"))
    return data["chunks"]


def load_chunk_meta(config) -> dict:
    return json.loads(config.CHUNKS_JSON.read_text(encoding="utf-8")).get("meta", {})


def save_embeddings(path, ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids), vectors=vectors.astype(np.float32))


def load_embeddings(path):
    """(ids, matrix) or None when the file does not exist."""
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return [str(i) for i in data["ids"]], data["vectors"]


def rag_files_status(config) -> dict:
    """What exists, without loading anything: used by /api/stats."""
    return {"chunks": config.CHUNKS_JSON.exists(), "embeddings": config.EMBEDDINGS_NPZ.exists(),
            "key": load_api_key(config) is not None}


def build_service(config, engine, client: GeminiClient | None = None):
    """RagService from the files on disk. FileNotFoundError when chunks.json is missing,
    PermissionError when there is no API key."""
    from .answer import RagService
    from .retriever import HybridRetriever
    chunks = load_chunks(config)                             # FileNotFoundError if scripts/12 was not run
    meta = load_chunk_meta(config)
    if meta.get("mode") != engine.analyzer.mode:
        raise ValueError(f"chunks.json was built with mode {meta.get('mode')!r} but the search uses "
                         f"{engine.analyzer.mode!r}: run scripts/12_build_chunks.py again")
    if client is None:
        key = load_api_key(config)
        if not key:
            raise PermissionError("No Gemini API key")
        client = GeminiClient(key, config)
    retriever = HybridRetriever(engine.analyzer, chunks, load_embeddings(config.EMBEDDINGS_NPZ), client,
                                config.LOG_BASE, config.RAG_CANDIDATES, config.RRF_K, config.RAG_MAX_PER_DOC)
    return RagService(retriever, client, engine.documents, config.RAG_TOP_PASSAGES)
