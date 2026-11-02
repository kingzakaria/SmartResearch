"""Duplicates and vocabulary zones. Run: pytest tests/test_duplicates_vocabulary.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.duplicates import (  # noqa: E402
    documents_to_drop, group_duplicates, jaccard_pairs, select_duplicate_pairs)
from src.indexing.inverted_index import InvertedIndex  # noqa: E402
from src.indexing.vocabulary import FREQUENT, KEPT, RARE, classify_terms, prune, summary  # noqa: E402

TERMS = {
    "1": ["شفع", "بيع", "عقار", "مزاد"],
    "2": ["شفع", "بيع", "عقار", "مزاد", "طعن"],     # copy of 1 with one extra term
    "3": ["حيز", "هادئ", "علن"],
    "4": ["شفع", "تحفيظ"],
}


def test_jaccard_pairs_finds_the_copy_and_sorts():
    pairs = jaccard_pairs(TERMS, min_score=0.5)
    assert pairs[0][:2] == ("1", "2") and pairs[0][2] == 4 / 5
    assert all(p[2] >= 0.5 for p in pairs) and "3" not in {x for p in pairs for x in p[:2]}


def test_groups_are_transitive_and_threshold_matters():
    pairs = [("1", "2", 0.9), ("2", "5", 0.85), ("7", "8", 0.6)]
    assert group_duplicates(pairs, 0.8) == [["1", "2", "5"]]
    assert group_duplicates(pairs, 0.5) == [["1", "2", "5"], ["7", "8"]]


def test_the_longest_copy_is_kept():
    docs = {"1": {"n_words": 100}, "2": {"n_words": 120}, "3": {"n_words": 120}}
    assert documents_to_drop([["1", "2", "3"]], docs) == ["1", "3"]


def test_zones_cut_both_ends():
    docs = {str(i): ["le", "mot"] + (["sujet"] if i < 5 else []) + (["nom"] if i == 0 else []) for i in range(10)}
    index = InvertedIndex.build(docs)
    zones = classify_terms(index, min_df=2, max_ratio=0.8)
    assert zones == {"le": FREQUENT, "mot": FREQUENT, "sujet": KEPT, "nom": RARE}
    assert prune(docs, zones)["0"] == ["sujet"] and prune(docs, zones)["9"] == []
    assert prune(docs, zones, drop=(FREQUENT,))["0"] == ["sujet", "nom"]
    s = summary(index, zones)
    assert s[FREQUENT]["terms"] == 2 and s[FREQUENT]["tokens"] == 20 and s[KEPT]["tokens"] == 5


def test_same_template_but_different_cases_is_not_a_duplicate():
    docs = {"1": {"case_no": "2022/4/1/5662"}, "2": {"case_no": "2022/4/1/5662"},
            "3": {"case_no": "2022/4/1/5663"}, "4": {"case_no": None}}
    pairs = [("1", "2", 0.95), ("1", "3", 0.90), ("3", "4", 0.98), ("2", "4", 0.9)]
    assert select_duplicate_pairs(pairs, docs, 0.8, 0.97) == [("1", "2", 0.95), ("3", "4", 0.98)]
