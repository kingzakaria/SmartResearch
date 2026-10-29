"""TF-IDF cosine search over analyzed documents."""

from collections.abc import Mapping
from typing import Any

from config import DEFAULT_TOP_K
from src.indexing.tfidf import TfidfIndex
from src.morphology.analyzer import ArabicMorphologyAnalyzer
from src.similarity.measures import cosine_similarity


class SearchEngine:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        analyzer: ArabicMorphologyAnalyzer,
    ) -> None:
        self.documents = {doc["doc_id"]: doc for doc in documents}
        self.analyzer = analyzer
        self.terms = {
            doc_id: analyzer.analyze(str(doc.get("text", "")))
            for doc_id, doc in self.documents.items()
        }
        self.index = TfidfIndex().fit(self.terms)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = self.index.transform(self.analyzer.analyze(query))
        if not query_vector:
            return []
        ranked = sorted(
            (
                (doc_id, cosine_similarity(query_vector, vector))
                for doc_id, vector in self.index.vectors.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            {"doc_id": doc_id, "score": score, "document": self.documents[doc_id]}
            for doc_id, score in ranked[:top_k]
            if score > 0
        ]
