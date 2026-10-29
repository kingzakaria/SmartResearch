"""Standard ranked-search evaluation metrics."""

import math
from collections.abc import Iterable


def precision_at_k(ranked: Iterable[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    results = list(ranked)[:k]
    return sum(doc_id in relevant for doc_id in results) / k


def recall_at_k(ranked: Iterable[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    return sum(doc_id in relevant for doc_id in list(ranked)[:k]) / len(relevant) if relevant else 0.0


def average_precision(ranked: Iterable[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg_at_k(ranked: Iterable[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    results = list(ranked)[:k]
    dcg = sum(1 / math.log2(rank + 1) for rank, doc_id in enumerate(results, start=1) if doc_id in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
