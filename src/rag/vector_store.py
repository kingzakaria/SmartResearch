"""Small TF-IDF vector store for persisted local passage retrieval."""

import pickle
from pathlib import Path
from typing import Any

from src.rag.embedder import TfidfEmbedder
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.embedder = TfidfEmbedder().fit([str(chunk["text"]) for chunk in chunks])
        self.matrix = self.embedder.encode([str(chunk["text"]) for chunk in chunks])

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query.strip() or not self.chunks:
            return []
        query_vector = self.embedder.encode([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        return [
            {**self.chunks[index], "score": float(score)}
            for index, score in ranked[:top_k]
            if score > 0
        ]

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path: Path | str) -> "VectorStore":
        with Path(path).open("rb") as stream:
            store = pickle.load(stream)
        if not isinstance(store, cls):
            raise TypeError("vector store file contains an unexpected object")
        return store
