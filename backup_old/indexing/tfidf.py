"""Sparse TF-IDF weights using base-10 inverse document frequency."""

import math
from collections import Counter
from collections.abc import Iterable, Mapping


class TfidfIndex:
    def __init__(self) -> None:
        self.document_count = 0
        self.document_frequency: dict[str, int] = {}
        self.vectors: dict[str, dict[str, float]] = {}

    def fit(self, documents: Mapping[str, Iterable[str]]) -> "TfidfIndex":
        materialized = {doc_id: list(terms) for doc_id, terms in documents.items()}
        self.document_count = len(materialized)
        frequencies: Counter[str] = Counter()
        for terms in materialized.values():
            frequencies.update(set(terms))
        self.document_frequency = dict(frequencies)
        self.vectors = {doc_id: self.transform(terms) for doc_id, terms in materialized.items()}
        return self

    def transform(self, terms: Iterable[str]) -> dict[str, float]:
        counts = Counter(terms)
        total = sum(counts.values())
        if not total or not self.document_count:
            return {}
        vector = {
            term: (frequency / total) * math.log10(self.document_count / self.document_frequency[term])
            for term, frequency in counts.items()
            if self.document_frequency.get(term, 0) > 0
        }
        return vector
