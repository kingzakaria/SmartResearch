"""Similarity and distance measures for sparse term vectors."""

import math
from collections.abc import Mapping, Iterable


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    return dot / (left_norm * right_norm)


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def dice_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0


def euclidean_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = left.keys() | right.keys()
    return math.sqrt(sum((left.get(key, 0.0) - right.get(key, 0.0)) ** 2 for key in keys))


def manhattan_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = left.keys() | right.keys()
    return sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
