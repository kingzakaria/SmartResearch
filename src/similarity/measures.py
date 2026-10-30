"""Similarity and distance measures of the course, written by hand.

Vectors are SPARSE dictionaries {term: weight} (a missing term has weight 0).
Sets are any iterable of terms.

SIMILARITIES (higher = closer)         DISTANCES (lower = closer)
  cosine   : angle between vectors       euclidean : straight-line distance
  jaccard  : |A ∩ B| / |A ∪ B|           manhattan : sum of absolute differences
  dice     : 2|A ∩ B| / (|A| + |B|)
"""
import math


def dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a                                   # iterate over the smaller vector
    return sum(w * b[t] for t, w in a.items() if t in b)


def norm(a: dict) -> float:
    return math.sqrt(sum(w * w for w in a.values()))


def cosine(a: dict, b: dict) -> float:
    """(a . b) / (||a|| ||b||). Ignores the length of the documents. 0 if a vector is empty."""
    na, nb = norm(a), norm(b)
    return dot(a, b) / (na * nb) if na and nb else 0.0


def euclidean(a: dict, b: dict) -> float:
    """sqrt( sum (a_i - b_i)^2 ) over the union of the terms."""
    total = 0.0
    for t in set(a) | set(b):
        total += (a.get(t, 0.0) - b.get(t, 0.0)) ** 2
    return math.sqrt(total)


def manhattan(a: dict, b: dict) -> float:
    """sum |a_i - b_i| over the union of the terms."""
    return sum(abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in set(a) | set(b))


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def dice(a, b) -> float:
    a, b = set(a), set(b)
    total = len(a) + len(b)
    return 2 * len(a & b) / total if total else 0.0


# name -> (function, works on, is a distance)
MEASURES = {
    "cosine": (cosine, "vectors", False),
    "jaccard": (jaccard, "sets", False),
    "dice": (dice, "sets", False),
    "euclidean": (euclidean, "vectors", True),
    "manhattan": (manhattan, "vectors", True),
}
