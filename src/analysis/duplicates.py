"""Near-duplicate detection.

Some decisions are published twice (same text, sometimes a different header). If one copy lands in the
training part of a classifier and the other in the test part, the test is not honest: the model has
already seen the answer. So duplicates are found first and kept together (or only one copy is used).

Two documents are compared by the Jaccard similarity of their TERM SETS (the stems after the morphology):
    J(a, b) = |a ∩ b| / |a ∪ b|        1.0 = same vocabulary, 0 = nothing in common
"""


def _jaccard_pairs_sparse(sets: dict, min_score: float):
    """Same result as the loops below, with a sparse matrix product (shared terms of ALL pairs at once).
    None when numpy/scipy are not installed (the loops are used instead)."""
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
    except ImportError:
        return None
    ids = list(sets)
    vocabulary: dict[str, int] = {}
    rows, cols = [], []
    for i, doc_id in enumerate(ids):
        for term in sets[doc_id]:
            rows.append(i)
            cols.append(vocabulary.setdefault(term, len(vocabulary)))
    matrix = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(ids), len(vocabulary)))
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    pairs = []
    step = 500                                           # blocks of rows keep the memory small
    for start in range(0, len(ids), step):
        block = (matrix[start:start + step] @ matrix.T).toarray()
        for r in range(block.shape[0]):
            i = start + r
            common = block[r, i + 1:]
            union = sizes[i] + sizes[i + 1:] - common
            score = np.divide(common, union, out=np.zeros_like(common), where=union > 0)
            for off in np.nonzero(score >= min_score)[0]:
                j = i + 1 + int(off)
                a, b = ids[i], ids[j]
                pairs.append((a, b, float(score[off])) if a < b else (b, a, float(score[off])))
    pairs.sort(key=lambda p: (-p[2], _order(p[0]), _order(p[1])))
    return pairs


def jaccard_pairs(terms_by_doc: dict, min_score: float = 0.5) -> list[tuple[str, str, float]]:
    """All pairs of documents whose Jaccard similarity is at least `min_score`, most similar first.

    An inverted list (term -> documents) is used so that only documents sharing a term are compared."""
    sets = {str(d): set(t) for d, t in terms_by_doc.items()}
    if len(sets) > 800:                                  # thousands of decisions: pairwise loops are too slow
        fast = _jaccard_pairs_sparse(sets, min_score)
        if fast is not None:
            return fast
    postings: dict[str, list[str]] = {}
    for doc_id, terms in sets.items():
        for term in terms:
            postings.setdefault(term, []).append(doc_id)

    shared: dict[tuple[str, str], int] = {}
    for docs in postings.values():
        for i, a in enumerate(docs):
            for b in docs[i + 1:]:
                key = (a, b) if a < b else (b, a)
                shared[key] = shared.get(key, 0) + 1

    pairs = []
    for (a, b), common in shared.items():
        union = len(sets[a]) + len(sets[b]) - common
        score = common / union if union else 0.0
        if score >= min_score:
            pairs.append((a, b, score))
    pairs.sort(key=lambda p: (-p[2], _order(p[0]), _order(p[1])))
    return pairs


def group_duplicates(pairs, threshold: float = 0.8) -> list[list[str]]:
    """Groups of documents linked by pairs with score >= threshold (union-find: if A~B and B~C then A, B, C)."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, score in pairs:
        if score >= threshold:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    return sorted((sorted(g, key=_order) for g in groups.values()), key=lambda g: _order(g[0]))


def select_duplicate_pairs(pairs, documents: dict, threshold: float = 0.8, strict: float = 0.97):
    """Keep the pairs that are really the SAME decision published twice:
    the same case number on both documents and score >= threshold, or an almost identical text (>= strict).
    Two different decisions that merely follow the same template (different case numbers) are not duplicates."""
    chosen = []
    for a, b, score in pairs:
        case_a, case_b = documents.get(a, {}).get("case_no"), documents.get(b, {}).get("case_no")
        if (case_a and case_a == case_b and score >= threshold) or score >= strict:
            chosen.append((a, b, score))
    return chosen


def documents_to_drop(groups, documents: dict) -> list[str]:
    """In each group keep the document with the most words (ties: the smallest id); the others are dropped."""
    drop = []
    for group in groups:
        keep = max(group, key=lambda d: (documents.get(d, {}).get("n_words") or 0, -_order(d)[0]))
        drop += [d for d in group if d != keep]
    return sorted(drop, key=_order)


def _order(doc_id: str):
    return (0, int(doc_id)) if str(doc_id).isdigit() else (1, 0)
