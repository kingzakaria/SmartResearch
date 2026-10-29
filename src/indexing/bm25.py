"""Okapi BM25 scoring over tokenized documents."""

import math
from collections import Counter
from collections.abc import Iterable, Mapping

from config import DEFAULT_BM25_B, DEFAULT_BM25_K1


class BM25Index:
    def __init__(
        self,
        documents: Mapping[str, Iterable[str]],
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be between 0 and 1")
        self.documents = {key: Counter(value) for key, value in documents.items()}
        self.k1, self.b = k1, b
        self.lengths = {key: sum(value.values()) for key, value in self.documents.items()}
        self.average_length = sum(self.lengths.values()) / max(len(self.lengths), 1)
        self.document_frequency = Counter(
            term for counts in self.documents.values() for term in counts
        )

    def score(self, query_terms: Iterable[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        total_docs = len(self.documents)
        unique_terms = set(query_terms)
        for doc_id, counts in self.documents.items():
            score = 0.0
            for term in unique_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self.document_frequency[term]
                idf = math.log1p((total_docs - df + 0.5) / (df + 0.5))
                length_norm = 1 - self.b + self.b * self.lengths[doc_id] / max(self.average_length, 1)
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
            scores[doc_id] = score
        return scores
