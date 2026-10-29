"""Filter terms by minimum document frequency."""

from collections.abc import Iterable, Mapping


def filter_by_document_frequency(
    documents: Mapping[str, Iterable[str]], minimum_frequency: int
) -> dict[str, list[str]]:
    if minimum_frequency < 1:
        raise ValueError("minimum_frequency must be at least 1")
    materialized = {doc_id: list(terms) for doc_id, terms in documents.items()}
    frequencies: dict[str, int] = {}
    for terms in materialized.values():
        for term in set(terms):
            frequencies[term] = frequencies.get(term, 0) + 1
    return {
        doc_id: [term for term in terms if frequencies[term] >= minimum_frequency]
        for doc_id, terms in materialized.items()
    }
