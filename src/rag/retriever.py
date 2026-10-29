"""Reciprocal-rank fusion of BM25 and vector retrieval."""

from collections.abc import Mapping, Iterable
from config import DEFAULT_RRF_RANK_CONSTANT
from src.indexing.bm25 import BM25Index
from src.rag.vector_store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        documents: Mapping[str, Iterable[str]],
        vector_store: VectorStore,
        rank_constant: int = DEFAULT_RRF_RANK_CONSTANT,
    ) -> None:
        if rank_constant < 1:
            raise ValueError("rank_constant must be positive")
        self.lexical = BM25Index(documents)
        self.vector_store = vector_store
        self.rank_constant = rank_constant

    def retrieve(self, query: str, query_terms: list[str], top_k: int = 5) -> list[dict[str, object]]:
        lexical = sorted(self.lexical.score(query_terms).items(), key=lambda item: (-item[1], item[0]))
        lexical = [(doc_id, score) for doc_id, score in lexical if score > 0][:top_k]
        vector = self.vector_store.search(query, max(top_k * 3, top_k))
        fused: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(lexical, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (self.rank_constant + rank)
        chunk_by_doc = {}
        for rank, chunk in enumerate(vector, start=1):
            doc_id = str(chunk["doc_id"])
            fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (self.rank_constant + rank)
            chunk_by_doc.setdefault(doc_id, chunk)
        return [
            {"doc_id": doc_id, "score": score, "chunk": chunk_by_doc.get(doc_id)}
            for doc_id, score in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        ]
