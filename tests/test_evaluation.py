import pytest

from src.evaluation.judgments import parse_relevance, read_compact, read_pool, write_compact
from src.evaluation.metrics import average_precision, f_measure, precision_at_k, recall_at_k, reciprocal_rank

RANKED = ["a", "b", "c", "d", "e"]
REL = {"a", "c", "z"}   # z is relevant but never retrieved


def test_precision_and_recall():
    assert precision_at_k(RANKED, REL, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RANKED, REL, 1) == 1.0
    assert recall_at_k(RANKED, REL, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, set(), 5) == 0.0


def test_precision_short_list_not_rewarded():
    assert precision_at_k(["a"], {"a"}, 10) == pytest.approx(0.1)


def test_average_precision_hand_computed():
    # hits at rank 1 (1/1) and rank 3 (2/3); the third relevant one is never found -> (1 + 2/3) / 3
    assert average_precision(RANKED, REL) == pytest.approx((1 + 2 / 3) / 3)
    assert average_precision(RANKED, set()) == 0.0
    assert average_precision(["x", "y"], {"a"}) == 0.0


def test_reciprocal_rank_and_f():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5
    assert reciprocal_rank(["x"], {"a"}) == 0.0
    assert f_measure(0.5, 0.5) == 0.5 and f_measure(0, 0) == 0.0


def test_parse_relevance():
    assert parse_relevance("1") == 1 and parse_relevance(" Yes ") == 1 and parse_relevance("0") == 0
    assert parse_relevance("") is None and parse_relevance(None) is None
    with pytest.raises(ValueError):
        parse_relevance("maybe")


def test_read_pool_semicolon_and_bom(tmp_path):
    p = tmp_path / "pool.csv"
    p.write_text("query_id;query;doc_id;relevant\nq01;شفعة;12;1\nq01;شفعة;ps-3;\n", encoding="utf-8-sig")
    rows = read_pool(p)
    assert [r["relevant"] for r in rows] == [1, None] and rows[1]["doc_id"] == "ps-3"


def test_read_pool_legacy_encoding_is_reported(tmp_path):
    p = tmp_path / "pool.csv"
    p.write_bytes("query_id,query,doc_id,relevant\nq01,???? ????,12,1\n".encode("utf-8"))
    with pytest.raises(ValueError):
        read_pool(p)


def test_compact_roundtrip(tmp_path):
    p = tmp_path / "eval" / "judgments.csv"
    write_compact(p, {("q01", "12"): 1, ("q01", "ps-3"): 0})
    assert read_compact(p) == {("q01", "12"): 1, ("q01", "ps-3"): 0}
    assert read_compact(tmp_path / "none.csv") == {}
