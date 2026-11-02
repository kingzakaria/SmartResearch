"""K-Means on TF-IDF vectors, written by hand (spherical K-Means: the similarity is the cosine).

    1. choose k starting centroids (k-means++: far from each other)
    2. assign every document to the centroid with the highest cosine
    3. move each centroid to the (normalised) mean of its documents
    4. repeat 2-3 until nothing changes; run several starts and keep the most cohesive result

Vectors are sparse dictionaries {term: weight}, as everywhere in the project.
"""
import math
import random


def norm(v: dict) -> float:
    return math.sqrt(sum(w * w for w in v.values()))


def unit(v: dict) -> dict:
    n = norm(v)
    return {t: w / n for t, w in v.items()} if n else {}


def dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b[t] for t, w in a.items() if t in b)


def _mean_direction(vectors) -> dict:
    total: dict[str, float] = {}
    for v in vectors:
        for t, w in v.items():
            total[t] = total.get(t, 0.0) + w
    return unit(total)


def _init_plus_plus(units: dict, ids: list, k: int, rng: random.Random) -> list[dict]:
    centroids = [units[rng.choice(ids)]]
    while len(centroids) < k:
        weights = [max(0.0, 1.0 - max(dot(units[d], c) for c in centroids)) ** 2 for d in ids]
        pick = rng.choices(ids, weights=weights)[0] if sum(weights) > 0 else rng.choice(ids)
        centroids.append(units[pick])
    return centroids


def kmeans(vectors: dict, k: int, seed: int = 42, n_init: int = 10, max_iter: int = 100):
    """Returns (assignments {doc_id: cluster}, centroids [unit vectors], cohesion).
    cohesion = mean cosine between a document and its centroid (higher = tighter groups)."""
    ids = sorted(vectors)
    if not 1 <= k <= len(ids):
        raise ValueError(f"k must be between 1 and the number of documents ({len(ids)})")
    units = {d: unit(vectors[d]) for d in ids}
    best = None
    for run in range(n_init):
        rng = random.Random(seed + run)
        centroids = _init_plus_plus(units, ids, k, rng)
        assign: dict = {}
        for _ in range(max_iter):
            new = {d: max(range(k), key=lambda c: (dot(units[d], centroids[c]), -c)) for d in ids}
            if new == assign:
                break
            assign = new
            for c in range(k):
                members = [units[d] for d in ids if assign[d] == c]
                if members:
                    centroids[c] = _mean_direction(members)
                else:                                   # empty cluster: restart it on the worst-placed document
                    worst = min(ids, key=lambda d: dot(units[d], centroids[assign[d]]))
                    centroids[c] = units[worst]
        cohesion = sum(dot(units[d], centroids[assign[d]]) for d in ids) / len(ids)
        if best is None or cohesion > best[2] + 1e-12:
            best = (assign, centroids, cohesion)
    return best


def similarity_matrix(vectors: dict) -> tuple[list, list[list[float]]]:
    ids = sorted(vectors)
    units = [unit(vectors[d]) for d in ids]
    n = len(ids)
    sims = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sims[i][j] = sims[j][i] = dot(units[i], units[j])
    return ids, sims


def silhouette(ids: list, sims: list[list[float]], assign: dict) -> float:
    """Mean silhouette with cosine distance (1 - cos): near 1 = well separated groups, near 0 = overlapping."""
    labels = [assign[d] for d in ids]
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(labels):
        groups.setdefault(c, []).append(i)
    if len(groups) < 2:
        return 0.0
    total = 0.0
    for i, c in enumerate(labels):
        own = [j for j in groups[c] if j != i]
        if not own:
            continue                                     # a group of one scores 0
        a = sum(1 - sims[i][j] for j in own) / len(own)
        b = min(sum(1 - sims[i][j] for j in members) / len(members)
                for other, members in groups.items() if other != c)
        total += (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return total / len(ids)
