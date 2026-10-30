"""Levenshtein (edit) distance and spelling suggestions for query terms.

The distance is the minimum number of insertions, deletions and substitutions that turn one word
into another. It is used to propose "did you mean ..." when a query term is not in the index
(typo, OCR error, spelling variant).
"""


def levenshtein(a: str, b: str, max_distance: int | None = None) -> int:
    """Edit distance. With max_distance, stops early and returns max_distance + 1 when exceeded."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a                                    # a is the longer word
    if max_distance is not None and len(a) - len(b) > max_distance:
        return max_distance + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1,                    # deletion
                               current[j - 1] + 1,                 # insertion
                               previous[j - 1] + (ca != cb)))      # substitution
        if max_distance is not None and min(current) > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """Distance turned into a similarity in [0, 1]."""
    longest = max(len(a), len(b))
    return 1 - levenshtein(a, b) / longest if longest else 1.0


def suggest(term: str, vocabulary, max_distance: int | None = None, k: int = 3, df=None) -> list[tuple[str, int]]:
    """The k closest vocabulary terms as (term, distance), nearest first.
    Short terms tolerate 1 error, longer ones 2. Between equally close terms, the most frequent
    one (highest df, if a df function is given) comes first."""
    if max_distance is None:
        max_distance = 1 if len(term) <= 4 else 2
    found = []
    for candidate in vocabulary:
        if abs(len(candidate) - len(term)) > max_distance:
            continue
        distance = levenshtein(term, candidate, max_distance)
        if 0 < distance <= max_distance:
            found.append((distance, -(df(candidate) if df else 0), candidate))
    found.sort()
    return [(candidate, distance) for distance, _, candidate in found[:k]]
