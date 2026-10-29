"""Chi-square feature scoring for labeled term documents."""

from collections import Counter
from collections.abc import Iterable, Mapping


def chi2_scores(
    documents: Mapping[str, Iterable[str]], labels: Mapping[str, str]
) -> dict[str, dict[str, float]]:
    doc_ids = [doc_id for doc_id in documents if doc_id in labels]
    classes = sorted({labels[doc_id] for doc_id in doc_ids})
    term_documents: dict[str, set[str]] = {}
    for doc_id in doc_ids:
        for term in set(documents[doc_id]):
            term_documents.setdefault(term, set()).add(doc_id)
    result: dict[str, dict[str, float]] = {}
    total = len(doc_ids)
    for term, containing in term_documents.items():
        scores = {}
        for label in classes:
            positive = sum(labels[doc_id] == label for doc_id in containing)
            class_total = sum(labels[doc_id] == label for doc_id in doc_ids)
            expected = class_total * len(containing) / total if total else 0
            scores[label] = (positive - expected) ** 2 / expected if expected else 0.0
        scores["chi2max"] = max(scores.values(), default=0.0)
        result[term] = scores
    return result
