"""Labels, chi-square, purity and adjusted Rand index. Run: pytest tests/test_labels.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.labels import adjusted_rand_index, chi_square, prefix_of, purity, top_terms_by_class  # noqa: E402


def test_prefix_of():
    assert prefix_of("2/87") == "2" and prefix_of("4/06") == "4"
    assert prefix_of("130") is None and prefix_of(None) is None and prefix_of("x/12") is None


def test_chi_square_of_a_known_table():
    # 100 docs, class of 40, term in 40 docs, all of them in the class: perfect link
    assert chi_square(100, 40, 40, 40) == pytest.approx(100 * (40 * 60) ** 2 / (40 * 60 * 40 * 60))
    assert chi_square(100, 50, 50, 25) == 0                       # independent
    assert chi_square(100, 40, 40, 0) < 0                         # the term avoids the class


def test_top_terms_find_the_vocabulary_of_each_class():
    terms = {str(i): ["شفع", "بيع", "عام"] for i in range(5)} | {str(i): ["تحفيظ", "تعرض", "عام"] for i in range(5, 10)}
    labels = {str(i): "a" if i < 5 else "b" for i in range(10)}
    top = top_terms_by_class(terms, labels, n=2, min_df=2)
    assert {t for t, _ in top["a"]} == {"شفع", "بيع"} and {t for t, _ in top["b"]} == {"تحفيظ", "تعرض"}


def test_documents_without_label_are_ignored():
    assert top_terms_by_class({"1": ["x"], "2": ["x"]}, {"1": None, "2": None}) == {}


def test_purity_and_adjusted_rand_index():
    labels = {str(i): "a" if i < 5 else "b" for i in range(10)}
    same = {d: 0 if l == "a" else 1 for d, l in labels.items()}
    assert purity(same, labels) == 1 and adjusted_rand_index(same, labels) == pytest.approx(1.0)
    mixed = {str(i): i % 2 for i in range(10)}
    assert purity(mixed, labels) == 0.6 and adjusted_rand_index(mixed, labels) < 0.1
