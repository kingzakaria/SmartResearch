"""Summarize cluster sizes, representative terms, and sample documents."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping


def inspect_clusters(
    documents: Mapping[str, Iterable[str]],
    labels: Mapping[str, int],
    top_terms: int = 10,
    sample_size: int = 3,
) -> dict[int, dict[str, object]]:
    members: dict[int, list[str]] = defaultdict(list)
    term_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for doc_id, label in labels.items():
        if doc_id not in documents:
            continue
        members[label].append(doc_id)
        term_counts[label].update(documents[doc_id])
    return {
        label: {
            "size": len(ids),
            "top_terms": [term for term, _ in term_counts[label].most_common(top_terms)],
            "sample_doc_ids": sorted(ids)[:sample_size],
        }
        for label, ids in sorted(members.items())
    }
