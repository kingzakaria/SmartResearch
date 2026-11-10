"""Naive Bayes (hand-written) and the chamber router. Run: pytest tests/test_naive_bayes.py"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classification.naive_bayes import NaiveBayes  # noqa: E402
from src.search.chamber_router import ChamberRouter  # noqa: E402

# the movie-review example of the course (Apprentissage supervise) and of the TP
REVIEWS = [("i loved the movie", "+"), ("i hated the movie", "-"), ("a great movie good movie", "+"),
           ("poor acting", "-"), ("great acting a good movie", "+")]


@pytest.fixture
def course_model():
    return NaiveBayes().fit([t.split() for t, _ in REVIEWS], [c for _, c in REVIEWS])


def test_course_example_counts_and_priors(course_model):
    m = course_model
    assert (m.m, m.totals["+"], m.totals["-"]) == (10, 14, 6)
    assert m.prior("+") == pytest.approx(0.6) and m.prior("-") == pytest.approx(0.4)
    assert m.p_word("movie", "+") == pytest.approx(5 / 24)             # 0.2083 in the course
    assert m.p_word("hated", "+") == pytest.approx(1 / 24)             # 0.0417: unseen word, Laplace smoothing


def test_course_example_scores_and_decision(course_model):
    tokens = "i hated the poor acting".split()
    scores = course_model.log_scores(tokens)
    assert math.exp(scores["+"]) == pytest.approx(6.03e-7, rel=1e-2)   # values of the course
    assert math.exp(scores["-"]) == pytest.approx(1.22e-5, rel=1e-2)
    assert course_model.predict(tokens) == "-"                          # negative, although "+" has the higher prior


def test_probabilities_of_a_class_sum_to_one(course_model):
    for c in course_model.classes:
        assert sum(course_model.p_word(w, c) for w in course_model.vocabulary) == pytest.approx(1.0)


def test_log_space_does_not_underflow():
    model = NaiveBayes().fit([["a", "b"], ["c", "d"]], ["x", "y"])
    long_text = ["a", "b"] * 2000                                       # the plain product would be 0.0
    proba = model.predict_proba(long_text)
    assert proba["x"] > 0.999 and sum(proba.values()) == pytest.approx(1.0)
    assert all(math.isfinite(s) for s in model.log_scores(long_text).values())


def test_unknown_words_are_ignored(course_model):
    assert course_model.log_scores(["zzz"]) == {c: math.log(course_model.prior(c)) for c in course_model.classes}
    assert course_model.n_known(["zzz", "movie"]) == 1


def test_needs_two_classes():
    with pytest.raises(ValueError):
        NaiveBayes().fit([["a"]], ["x"])
    with pytest.raises(ValueError):
        NaiveBayes().fit([["a"], ["b"]], ["x"])


def test_save_and_load_give_the_same_model(course_model, tmp_path):
    course_model.save(tmp_path / "nb.json", {"mode": "light"})
    loaded, meta = NaiveBayes.load(tmp_path / "nb.json")
    assert meta == {"mode": "light"} and loaded.m == course_model.m
    tokens = "i hated the poor acting".split()
    assert loaded.log_scores(tokens) == pytest.approx(course_model.log_scores(tokens))


# ---------- router ----------
@pytest.fixture
def router():
    model = NaiveBayes().fit([["شفع", "عقار"], ["شفع", "بيع"], ["طلق", "نفق"], ["ارث", "طلق"]], ["re", "re", "ps", "ps"])
    return ChamberRouter(model, 0.8)


def test_router_sends_a_clear_query_to_one_chamber(router):
    decision = router.route(["طلق", "نفق"])
    assert decision.predicted == "ps" and decision.searched == ["ps"] and decision.reason == "confident"


def test_router_searches_everything_when_no_word_is_known(router):
    decision = router.route(["كلمة"])
    assert decision.predicted is None and sorted(decision.searched) == ["ps", "re"] and decision.reason == "no_known_word"


def test_router_searches_everything_when_not_sure():
    model = NaiveBayes().fit([["شفع"], ["طلق"]], ["re", "ps"])
    router = ChamberRouter(model, 0.9)
    decision = router.route(["شفع", "طلق"])                              # one word for each chamber
    assert decision.reason == "low_confidence" and sorted(decision.searched) == ["ps", "re"]


def test_uniform_priors_remove_the_size_of_the_class():
    docs = [["a"]] * 9 + [["b"]]
    model = NaiveBayes().fit(docs + [["a", "b"]], ["big"] * 9 + ["small", "big"])
    empirical = model.predict_proba(["zzz"])                              # unknown word: only the prior speaks
    uniform = model.predict_proba(["zzz"], uniform_priors=True)
    assert empirical["big"] > 0.8 and uniform["big"] == pytest.approx(0.5)
    assert ChamberRouter(model, 0.8, uniform_priors=True).route(["zzz"]).reason == "no_known_word"
