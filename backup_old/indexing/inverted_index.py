"""Inverted index mapping terms to per-document term frequencies."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping


def build_inverted_index(
    documents: Mapping[str, Iterable[str]],
) -> dict[str, dict[str, int]]:
    index: dict[str, dict[str, int]] = defaultdict(dict)
    for doc_id, terms in documents.items():
        for term, frequency in Counter(terms).items():
            index[term][doc_id] = frequency
    return dict(index)


class InvertedIndex:
    def __init__(self, documents: Mapping[str, Iterable[str]]) -> None:
        self.documents = {doc_id: list(terms) for doc_id, terms in documents.items()}
        self.index = build_inverted_index(self.documents)

    def search(self, terms: Iterable[str]) -> set[str]:
        matches: set[str] = set()
        for term in terms:
            matches.update(self.index.get(term, {}))
        return matches
