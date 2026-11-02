"""Divide the collection into k groups (partitions) with K-Means.

Features used for the clustering are NOT the ones used for searching: they are cleaned more strongly
(the three-zone rule with a stricter upper limit, no bare numbers), so that the groups form around TOPICS and
not around the boilerplate phrases that almost every decision contains.
"""
from ..indexing.inverted_index import InvertedIndex
from ..indexing.tfidf import TfidfModel
from ..indexing.vocabulary import classify_terms, prune
from .kmeans import dot, kmeans, unit


def cluster_features(terms_by_doc: dict, drop=(), min_df: int = 2, max_ratio: float = 0.5, log_base: float = 10):
    """TF-IDF model on the cleaned vocabulary of the documents that are not dropped duplicates."""
    drop = {str(d) for d in drop}
    kept = {str(d): [t for t in terms if not t.isdigit()] for d, terms in terms_by_doc.items() if str(d) not in drop}
    zones = classify_terms(InvertedIndex.build(kept), min_df, max_ratio)
    return TfidfModel(InvertedIndex.build(prune(kept, zones)), log_base)


def build_partitions(terms_by_doc: dict, k: int = 3, drop=(), duplicate_groups=(), seed: int = 42,
                     min_df: int = 2, max_ratio: float = 0.5, log_base: float = 10, n_init: int = 10) -> dict:
    """The partition description (JSON-serialisable): assignments, centroids and what is needed to route a query."""
    drop = {str(d) for d in drop}
    model = cluster_features(terms_by_doc, drop, min_df, max_ratio, log_base)
    vectors = {d: model.doc_vector(d) for d in model.index.doc_len if model.norms[d] > 0}
    assign, centroids, cohesion = kmeans(vectors, k, seed=seed, n_init=n_init)
    assignments = {d: int(c) for d, c in assign.items()}

    sizes = [sum(1 for c in assignments.values() if c == i) for i in range(k)]
    for group in duplicate_groups:                      # a dropped copy goes where its kept twin is
        twin = next((d for d in map(str, group) if d not in drop and d in assignments), None)
        for d in map(str, group):
            if d in drop and twin:
                assignments[d] = assignments[twin]
    biggest = max(range(k), key=lambda c: sizes[c])
    for d in map(str, terms_by_doc):                    # documents without any usable feature
        assignments.setdefault(d, biggest)

    final_sizes = [sum(1 for c in assignments.values() if c == i) for i in range(k)]
    used = {t for c in centroids for t in c}
    return {
        "k": k, "seed": seed, "min_df": min_df, "max_ratio": max_ratio, "log_base": log_base,
        "cohesion": round(cohesion, 4), "sizes": final_sizes,
        "assignments": assignments,
        "centroids": [{t: round(w, 6) for t, w in c.items()} for c in centroids],
        "idf": {t: round(model.idf[t], 6) for t in used},
        "top_terms": [[t for t, _ in sorted(c.items(), key=lambda kv: -kv[1])[:15]] for c in centroids],
    }


def central_documents(partitions: dict, model_vectors: dict, n: int = 5) -> list[list[str]]:
    """For each group, the documents closest to its centroid (they best represent it)."""
    out = []
    for c in range(partitions["k"]):
        centroid = partitions["centroids"][c]
        members = [d for d, g in partitions["assignments"].items() if g == c and d in model_vectors]
        members.sort(key=lambda d: -dot(unit(model_vectors[d]), centroid))
        out.append(members[:n])
    return out
