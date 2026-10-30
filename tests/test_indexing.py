"""Inverted index and TF-IDF, checked against the numbers of the course.
Run: pytest tests/test_indexing.py
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.indexing.inverted_index import InvertedIndex  # noqa: E402
from src.indexing.tfidf import TfidfModel, idf, tf, weight  # noqa: E402

# 3 documents after preprocessing (the worked example used to explain the pipeline)
DOCS = {
    "d1": ["علم", "بيانات", "علم", "نص"],
    "d2": ["نص", "تنقيب", "نص"],
    "d3": ["رياضة", "كرة", "علم"],
}


# ---------- the formulas of the course ----------
def test_course_example_term_qui():
    """'qui' occurs 2 times in a 38-term document and is present in 2 of the 3 documents.
    TF = 2/38, IDF = log10(3/2): the exact value is 0.00927, shown as 0.0092 in the course."""
    assert tf(2, 38) == pytest.approx(2 / 38)
    assert idf(3, 2) == pytest.approx(math.log10(1.5))
    assert weight(2, 38, 3, 2) == pytest.approx(0.00927, abs=1e-5)
    assert abs(weight(2, 38, 3, 2) - 0.0092) < 1e-4


def test_absent_term_has_weight_zero():
    assert weight(0, 38, 3, 2) == 0


def test_idf_is_zero_for_a_term_in_every_document():
    assert idf(3, 3) == 0 and idf(358, 358) == 0


def test_log_base_is_a_parameter():
    assert idf(3, 2, base=math.e) == pytest.approx(math.log(1.5))
    assert idf(3, 2, base=10) != idf(3, 2, base=math.e)


# ---------- inverted index ----------
def test_postings_and_document_frequency():
    index = InvertedIndex.build(DOCS)
    assert index.postings["علم"] == {"d1": 2, "d3": 1}
    assert index.df("علم") == 2 and index.df("بيانات") == 1 and index.df("inconnu") == 0
    assert index.count("نص", "d2") == 2 and index.count("نص", "d3") == 0
    assert index.doc_len == {"d1": 4, "d2": 3, "d3": 3}
    assert index.n_docs == 3


def test_candidates_only_contain_documents_sharing_a_term():
    index = InvertedIndex.build(DOCS)
    assert index.candidates(["تنقيب"]) == {"d2"}
    assert index.candidates(["علم", "تنقيب"]) == {"d1", "d2", "d3"}
    assert index.candidates(["inconnu"]) == set()


def test_integer_doc_ids_are_stored_as_strings():
    index = InvertedIndex.build({8: ["علم"], 31: ["علم"]})
    assert index.df("علم") == 2 and index.count("علم", 8) == 1


def test_indexing_the_same_document_twice_is_an_error():
    index = InvertedIndex.build(DOCS)
    with pytest.raises(ValueError):
        index.add_document("d1", ["علم"])


def test_save_and_load_roundtrip(tmp_path):
    index = InvertedIndex.build(DOCS)
    index.save(tmp_path / "models" / "index.pkl")
    loaded = InvertedIndex.load(tmp_path / "models" / "index.pkl")
    assert loaded.postings == index.postings and loaded.doc_len == index.doc_len


# ---------- TF-IDF model and cosine ----------
def test_tfidf_matrix_of_the_worked_example():
    m = TfidfModel(InvertedIndex.build(DOCS))
    assert m.idf["علم"] == pytest.approx(0.1761, abs=1e-4)
    assert m.idf["بيانات"] == pytest.approx(0.4771, abs=1e-4)
    assert m.doc_vector("d1")["علم"] == pytest.approx(0.088, abs=1e-3)        # 0.5 x 0.176
    assert m.doc_vector("d1")["بيانات"] == pytest.approx(0.119, abs=1e-3)     # 0.25 x 0.477
    assert m.doc_vector("d2")["تنقيب"] == pytest.approx(0.159, abs=1e-3)      # 0.333 x 0.477


def test_cosine_ranking_of_the_worked_example():
    """Query 'علم نص': d1 (0.60) > d2 (0.42) > d3 (0.18)."""
    scores = TfidfModel(InvertedIndex.build(DOCS)).cosine_scores(["علم", "نص"])
    assert scores["d1"] == pytest.approx(0.60, abs=0.01)
    assert scores["d2"] == pytest.approx(0.42, abs=0.01)
    assert scores["d3"] == pytest.approx(0.18, abs=0.01)
    assert sorted(scores, key=scores.get, reverse=True) == ["d1", "d2", "d3"]


def test_repeating_the_query_does_not_change_the_scores():
    m = TfidfModel(InvertedIndex.build(DOCS))
    a, b = m.cosine_scores(["علم", "نص"]), m.cosine_scores(["علم", "نص"] * 3)
    assert {d: round(s, 9) for d, s in a.items()} == {d: round(s, 9) for d, s in b.items()}


def test_a_document_is_most_similar_to_itself():
    m = TfidfModel(InvertedIndex.build(DOCS))
    assert m.cosine(m.doc_vector("d1"), "d1") == pytest.approx(1.0)


def test_unknown_and_ubiquitous_terms_give_no_scores():
    m = TfidfModel(InvertedIndex.build({"a": ["x", "y"], "b": ["x", "z"]}))
    assert m.cosine_scores(["inconnu"]) == {}
    assert m.cosine(m.query_vector(["x"]), "a") == 0.0          # 'x' is in every document: IDF = 0


def test_top_terms_are_the_heaviest_ones():
    m = TfidfModel(InvertedIndex.build(DOCS))
    assert [t for t, _ in m.top_terms("d1", 2)] == ["بيانات", "علم"]


def test_build_metadata_survives_save_and_load(tmp_path):
    index = InvertedIndex.build(DOCS)
    index.meta = {"mode": "light", "log_base": 10}
    index.save(tmp_path / "index.pkl")
    assert InvertedIndex.load(tmp_path / "index.pkl").meta == {"mode": "light", "log_base": 10}
