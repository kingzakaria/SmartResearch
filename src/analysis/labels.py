"""Labels and how well two labellings agree.

prefix_of("2/87") -> "2": the first number of the decision number (the candidate category of the decision).
chi_square(): the chi-square statistic of the course for a term and a class (feature selection).
purity() and adjusted_rand_index(): how much a clustering agrees with an external labelling.
"""
import math
from collections import Counter


def prefix_of(decision_no) -> str | None:
    """'2/87' -> '2'. A decision number without '/' (e.g. '130') has no prefix."""
    if not decision_no or "/" not in str(decision_no):
        return None
    head = str(decision_no).split("/")[0].strip()
    return head if head.isdigit() else None


def chi_square(n_docs: int, in_class: int, with_term: int, both: int) -> float:
    """Chi-square of the 2x2 table (term present / absent) x (in class / not in class).
        a = both, b = with_term - both, c = in_class - both, d = the rest
        chi2 = N (ad - bc)^2 / ((a+b)(c+d)(a+c)(b+d))      high = the term and the class are linked
    The sign is positive when the term is MORE frequent in the class than elsewhere, negative otherwise."""
    a, b, c = both, with_term - both, in_class - both
    d = n_docs - a - b - c
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    if denominator == 0:
        return 0.0
    value = n_docs * (a * d - b * c) ** 2 / denominator
    return value if a * d - b * c > 0 else -value


def top_terms_by_class(terms_by_doc: dict, labels: dict, n: int = 12, min_df: int = 3) -> dict:
    """For each class, the terms most linked to it (positive chi-square), with their score."""
    docs = [d for d in terms_by_doc if labels.get(d) is not None]
    presence = {d: set(terms_by_doc[d]) for d in docs}
    df: Counter = Counter(t for d in docs for t in presence[d])
    class_size = Counter(labels[d] for d in docs)
    joint: dict = {}
    for d in docs:
        for t in presence[d]:
            joint[(t, labels[d])] = joint.get((t, labels[d]), 0) + 1
    out = {}
    for c in sorted(class_size):
        scored = [(t, chi_square(len(docs), class_size[c], df[t], joint.get((t, c), 0)))
                  for t in df if df[t] >= min_df]
        out[c] = sorted(((t, s) for t, s in scored if s > 0), key=lambda ts: -ts[1])[:n]
    return out


def purity(clusters: dict, labels: dict) -> float:
    """Share of documents whose cluster's majority label is their own label (1.0 = clusters never mix labels)."""
    docs = [d for d in clusters if labels.get(d) is not None]
    by_cluster: dict = {}
    for d in docs:
        by_cluster.setdefault(clusters[d], Counter())[labels[d]] += 1
    return sum(max(c.values()) for c in by_cluster.values()) / len(docs) if docs else 0.0


def _pairs(n: int) -> float:
    return n * (n - 1) / 2


def adjusted_rand_index(clusters: dict, labels: dict) -> float:
    """Agreement of two partitions corrected for chance: 1 = identical, about 0 = random, can be negative."""
    docs = [d for d in clusters if labels.get(d) is not None]
    n = len(docs)
    if n < 2:
        return 0.0
    table = Counter((clusters[d], labels[d]) for d in docs)
    rows = Counter(clusters[d] for d in docs)
    cols = Counter(labels[d] for d in docs)
    index = sum(_pairs(v) for v in table.values())
    sum_rows, sum_cols = sum(_pairs(v) for v in rows.values()), sum(_pairs(v) for v in cols.values())
    expected = sum_rows * sum_cols / _pairs(n)
    maximum = (sum_rows + sum_cols) / 2
    return (index - expected) / (maximum - expected) if maximum != expected else 1.0
