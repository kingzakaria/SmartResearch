"""Evaluation of a ranked list of results against the decisions judged relevant (course: precision, recall, MAP).

`ranked` is the list of doc_ids returned by a system, best first; `relevant` is the set of doc_ids judged relevant.
Documents that nobody judged count as not relevant (standard practice with pooling).
"""


def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    """Share of the k first results that are relevant. The denominator stays k even if fewer results come back
    (a system that returns 3 results for k = 10 is not rewarded for being short)."""
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    """Share of the relevant decisions that are in the k first results."""
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant) if relevant else 0.0


def average_precision(ranked: list, relevant: set) -> float:
    """Mean of the precision measured at the rank of each relevant result found; relevant decisions never retrieved
    count as 0. The mean of this value over the queries is the MAP."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def reciprocal_rank(ranked: list, relevant: set) -> float:
    """1 / rank of the first relevant result (0 if none): how quickly a user finds something useful."""
    for rank, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def f_measure(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
