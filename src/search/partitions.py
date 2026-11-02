"""Partitioned search: route the query to the best group(s) of documents and search only there.

    query -> cosine with the k centroids -> the n_probe closest groups -> inverted index of those groups only

Every group has its own inverted index (a subset of the main one). Documents are scored with the SAME TF-IDF model
as the full search, so a document gets the same score in both modes: the only difference is which documents
are looked at. A query is approximate when a relevant document sits in a group that was not searched.
"""
from ..clustering.kmeans import dot, unit


class Partitions:
    def __init__(self, data: dict, index, model):
        self.k = data["k"]
        self.sizes = data["sizes"]
        self.top_terms = data.get("top_terms", [])
        self.assignments = {str(d): int(c) for d, c in data["assignments"].items()}
        self.centroids = data["centroids"]
        self.idf = data["idf"]
        self.model = model
        members = [[d for d, c in self.assignments.items() if c == i] for i in range(self.k)]
        self.indexes = [index.subset(m) for m in members]

    def group_of(self, doc_id) -> int | None:
        return self.assignments.get(str(doc_id))

    def route(self, terms: list[str], n_probe: int = 1):
        """(groups to search, similarity of the query to each centroid, fallback).
        If none of the query terms belongs to the vocabulary used for clustering, no group can be
        preferred and every group is searched (fallback=True)."""
        everything = list(range(self.k))
        counts: dict[str, int] = {}
        for t in terms:
            if t in self.idf:
                counts[t] = counts.get(t, 0) + 1
        vector = unit({t: c / len(terms) * self.idf[t] for t, c in counts.items()})
        sims = [dot(vector, c) for c in self.centroids]
        if not vector or max(sims) <= 0:
            return everything, sims, True
        order = sorted(everything, key=lambda c: (-sims[c], c))
        return order[:max(1, min(n_probe, self.k))], sims, False

    def cosine_scores(self, terms: list[str], groups: list[int]) -> dict[str, float]:
        query_vector = self.model.query_vector(terms)
        found: set[str] = set()
        for g in groups:
            found |= self.indexes[g].candidates(list(query_vector))
        return {d: self.model.cosine(query_vector, d) for d in found}
