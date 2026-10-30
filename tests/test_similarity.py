"""Similarity measures and Levenshtein, checked against the worked example of the explanation.
Run: pytest tests/test_similarity.py
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.similarity import measures  # noqa: E402
from src.similarity.levenshtein import levenshtein, similarity, suggest  # noqa: E402

# vocabulary [python, data, java]
Q = {"python": 1, "data": 1}
A = {"python": 3, "data": 2}
B = {"data": 1, "java": 4}


def test_cosine():
    assert measures.cosine(Q, A) == pytest.approx(5 / math.sqrt(2 * 13))       # 0.98
    assert measures.cosine(Q, B) == pytest.approx(1 / math.sqrt(2 * 17))       # 0.17
    assert measures.cosine(Q, A) > measures.cosine(Q, B)


def test_cosine_ignores_document_length():
    assert measures.cosine(Q, {"python": 30, "data": 30}) == pytest.approx(1.0)


def test_cosine_of_empty_vector_is_zero():
    assert measures.cosine({}, A) == 0.0 and measures.cosine(Q, {"python": 0.0}) == 0.0


def test_euclidean_and_manhattan():
    assert measures.euclidean(Q, A) == pytest.approx(math.sqrt(5))             # 2.24
    assert measures.euclidean(Q, B) == pytest.approx(math.sqrt(17))            # 4.12
    assert measures.manhattan(Q, A) == 3 and measures.manhattan(Q, B) == 5
    assert measures.euclidean(Q, Q) == 0 and measures.manhattan(Q, Q) == 0


def test_jaccard_and_dice():
    d1 = {"python", "data", "analysis"}
    d2 = {"python", "data", "java", "web"}
    assert measures.jaccard(d1, d2) == pytest.approx(2 / 5)
    assert measures.dice(d1, d2) == pytest.approx(4 / 7)
    assert measures.jaccard([], []) == 0.0 and measures.dice([], []) == 0.0
    assert measures.jaccard(d1, d1) == 1.0


def test_measure_table_says_which_ones_are_distances():
    assert measures.MEASURES["euclidean"][2] and measures.MEASURES["manhattan"][2]
    assert not measures.MEASURES["cosine"][2]


# ---------- Levenshtein ----------
def test_levenshtein_known_values():
    assert levenshtein("pyhton", "python") == 2
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3 and levenshtein("abc", "abc") == 0


def test_levenshtein_on_arabic_spelling_variants():
    assert levenshtein("شفعه", "شفعة") == 1
    assert levenshtein("تحفيض", "تحفيظ") == 1


def test_levenshtein_early_exit():
    assert levenshtein("abcdef", "uvwxyz", max_distance=2) == 3
    assert levenshtein("abcdefgh", "ab", max_distance=2) == 3


def test_similarity_in_0_1():
    assert similarity("abc", "abc") == 1.0
    assert 0 <= similarity("abc", "xyz") <= 1


def test_suggest_prefers_the_closest_then_the_most_frequent():
    vocabulary = ["شفع", "شفعه", "تحفيظ", "عقار", "حيازه"]
    frequency = {"شفع": 10, "شفعه": 2}.get
    found = suggest("شفعا", vocabulary, df=lambda t: frequency(t, 0))
    assert found[0] == ("شفع", 1) and ("شفعه", 1) in found
    assert suggest("تحفيض", vocabulary)[0] == ("تحفيظ", 1)
    assert suggest("zzzzzz", vocabulary) == []
