"""Search engine on a small in-memory corpus. Run: pytest tests/test_search.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.indexing.inverted_index import InvertedIndex  # noqa: E402
from src.indexing.tfidf import TfidfModel  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402
from src.search.highlight import find_matches, make_snippet  # noqa: E402

# every body ends with "الفصل" so that this term is in all documents (IDF = 0)
BODIES = {
    1: "دعوى الشفعة تمسك الطاعن بالشفعة وبيع العقار بالمزاد الفصل",
    2: "مطلب التحفيظ والتعرض على مطلب التحفيظ العقاري الفصل",
    3: "دعوى الحيازة شروط الحيازة الهادئة العلنية الفصل",
    4: "الشفعة وتزاحم الشفعاء في بيع العقار الفصل",
    5: "الخبرة القضائية وتعليل القرار الاستئنافي الفصل",
    6: "دعوى التقادم وسقوط الدعوى بمرور المدة الفصل",
}
DOCS = [{"doc_id": i, "filename": f"{i}.txt", "decision_no": f"1/{i}", "date": "2023-01-01",
         "case_no": f"2020/1/1/{i}", "subject": f"موضوع {i}", "body": body} for i, body in BODIES.items()]


@pytest.fixture(scope="module")
def engine():
    analyzer = analyzer_from_config(config, "light")
    terms = {d["doc_id"]: analyzer.analyze(d["body"]) for d in DOCS}
    index = InvertedIndex.build(terms)
    return SearchEngine(analyzer, TfidfModel(index), DOCS)


def ids(response):
    return [r.doc_id for r in response.results]


# ---------- ranking ----------
def test_documents_with_the_term_are_found_and_sorted_by_score(engine):
    response = engine.search("الشفعة")
    assert set(ids(response)) == {"1", "4"}
    scores = [r.score for r in response.results]
    assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)
    assert response.results[0].rank == 1


def test_morphological_variants_of_the_query_give_the_same_ranking(engine):
    reference = ids(engine.search("الشفعة"))
    for variant in ("شفعة", "بالشفعة", "والشفعة", "الشفعه"):
        assert ids(engine.search(variant)) == reference


def test_result_carries_the_metadata(engine):
    r = engine.search("التحفيظ").results[0]
    assert r.doc_id == "2" and r.filename == "2.txt" and r.decision_no == "1/2"
    assert r.case_no == "2020/1/1/2" and r.subject == "موضوع 2" and r.matched_terms == ["تحفيظ"]


def test_k_limits_the_number_of_results(engine):
    response = engine.search("دعوى العقار", k=1)
    assert len(response.results) == 1 and response.total >= 3


# ---------- boolean AND ----------
def test_or_is_the_default_and_and_is_stricter(engine):
    assert set(ids(engine.search("الشفعة العقار"))) == {"1", "2", "4"}
    assert set(ids(engine.search("الشفعة العقار", require_all=True))) == {"1", "4"}


def test_and_with_an_unknown_term_returns_nothing(engine):
    assert engine.search("الشفعة سيارة", require_all=True).results == []         # سيارة is not in the corpus
    assert engine.search("الشفعة سيارة").results != []                           # OR still finds the rest


# ---------- unknown, ubiquitous and empty queries ----------
def test_unknown_term_gets_suggestions(engine):
    response = engine.search("شفعا")                          # typo of شفع
    assert response.results == [] and response.unknown == ["شفعا"]
    assert "شفع" in [term for term, _ in response.suggestions["شفعا"]]


def test_term_present_in_every_document_is_reported_and_scores_nothing(engine):
    response = engine.search("الفصل")
    assert response.results == [] and response.ignored == ["فصل"]


def test_stopword_only_query(engine):
    response = engine.search("في من على")
    assert response.terms == [] and response.results == []


# ---------- snippet ----------
def test_snippet_marks_the_original_words(engine):
    snippet = engine.search("الشفعة").results[0].snippet
    assert "«الشفعة»" in snippet and "«بالشفعة»" in snippet       # original spelling, not the stem


def test_find_matches_and_snippet_window():
    analyzer = analyzer_from_config(config, "light")
    text = "كلمة " * 100 + "الشفعة" + " كلمة" * 100
    spans = find_matches(text, ["شفع"], analyzer)
    assert len(spans) == 1 and text[spans[0][0]:spans[0][1]] == "الشفعة"
    snippet = make_snippet(text, spans, width=100)
    assert snippet.startswith("… ") and snippet.endswith(" …") and "«الشفعة»" in snippet
    assert len(snippet) < 100 + 60


def test_snippet_without_match_is_the_beginning_of_the_text():
    assert make_snippet("بداية النص " * 50, [], width=30).startswith("بداية النص")


# ---------- similar documents ----------
def test_similar_documents(engine):
    for measure in ("cosine", "jaccard", "dice"):
        assert engine.similar(1, k=1, measure=measure)[0][0] == "4"           # shares شفع, بيع, عقار
    distances = [value for _, value in engine.similar(1, k=5, measure="euclidean")]
    assert distances == sorted(distances)                                      # smaller = closer
    assert engine.similar(1, k=1, measure="manhattan")[0][0] == "4"


def test_similar_errors(engine):
    with pytest.raises(KeyError):
        engine.similar(999)
    with pytest.raises(ValueError):
        engine.similar(1, measure="euclid")


# ---------- loading from files, with the consistency check ----------
def _fake_config(tmp_path, mode_in_config):
    import json
    (tmp_path / "documents.json").write_text(json.dumps(DOCS, ensure_ascii=False), encoding="utf-8")
    return SimpleNamespace(INDEX_PKL=tmp_path / "index.pkl", DOCUMENTS_JSON=tmp_path / "documents.json",
                           STOPWORDS_GENERIC=config.STOPWORDS_GENERIC, STOPWORDS_LEGAL=config.STOPWORDS_LEGAL,
                           ANALYZER_MODE=mode_in_config, LOG_BASE=10)


def _save_index(path, mode):
    analyzer = analyzer_from_config(config, "light")
    index = InvertedIndex.build({d["doc_id"]: analyzer.analyze(d["body"]) for d in DOCS})
    index.meta = {"mode": mode}
    index.save(path)


def test_from_config_loads_a_consistent_index(tmp_path):
    cfg = _fake_config(tmp_path, "light")
    _save_index(cfg.INDEX_PKL, "light")
    assert set(ids(SearchEngine.from_config(cfg).search("الشفعة"))) == {"1", "4"}


def test_from_config_refuses_an_index_built_with_another_mode(tmp_path):
    cfg = _fake_config(tmp_path, "light")
    _save_index(cfg.INDEX_PKL, "root")
    with pytest.raises(ValueError, match="03_build_index"):
        SearchEngine.from_config(cfg)
