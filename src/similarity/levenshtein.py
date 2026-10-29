"""Levenshtein distance and nearest-vocabulary query correction."""

from config import DEFAULT_SPELLING_DISTANCE


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for col, right_char in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[col] + 1, previous[col - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def correct_query(
    query: str, vocabulary: set[str], max_distance: int = DEFAULT_SPELLING_DISTANCE
) -> str:
    corrected = []
    for token in query.split():
        if token in vocabulary or not vocabulary:
            corrected.append(token)
            continue
        best = min(vocabulary, key=lambda word: (levenshtein_distance(token, word), word))
        corrected.append(best if levenshtein_distance(token, best) <= max_distance else token)
    return " ".join(corrected)
