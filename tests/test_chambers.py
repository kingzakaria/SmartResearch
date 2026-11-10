"""Search over two chambers with Naive Bayes routing. Run: pytest tests/test_chambers.py"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from app.api import create_app  # noqa: E402
from src.analysis.duplicates import _jaccard_pairs_sparse, jaccard_pairs  # noqa: E402
from src.classification.naive_bayes import NaiveBayes  # noqa: E402
from src.clustering.partitioning import build_partitions  # noqa: E402
from src.indexing.inverted_index import InvertedIndex  # noqa: E402
from src.indexing.tfidf import TfidfModel  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402
from src.search.chamber_router import ChamberRouter  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402

BODIES = [
    (1, "real_estate", "دعوى الشفعة وبيع العقار بالمزاد"),
    (2, "real_estate", "مطلب التحفيظ والتعرض على مطلب التحفيظ العقاري"),
    (3, "real_estate", "دعوى الحيازة شروط الحيازة الهادئة العلنية"),
    (4, "real_estate", "الشفعة وتزاحم الشفعاء في بيع العقار"),
    ("ps-1", "personal_status", "دعوى الطلاق للشقاق والنفقة"),
    ("ps-2", "personal_status", "قسمة التركة والارث بين الورثة"),
    ("ps-3", "personal_status", "النفقة والحضانة بعد الطلاق"),
    ("ps-4", "personal_status", "اثبات النسب والارث والتركة"),
]
DOCS = [{"doc_id": i, "filename": f"{i}.txt", "decision_no": f"1/{i}", "date": "2023-01-01", "case_no": None,
         "subject": None, "body": b, "chamber": c} for i, c, b in BODIES]
NAMES = {"real_estate": "الغرفة العقارية", "personal_status": "غرفة الأحوال الشخصية"}


@pytest.fixture(scope="module")
def engine():
    analyzer = analyzer_from_config(config, "light")
    terms = {str(d["doc_id"]): analyzer.analyze(d["body"]) for d in DOCS}
    eng = SearchEngine(analyzer, TfidfModel(InvertedIndex.build(terms)), DOCS)
    eng.chamber_names = NAMES
    nb = NaiveBayes().fit([terms[str(d["doc_id"])] for d in DOCS], [d["chamber"] for d in DOCS])
    eng.router = ChamberRouter(nb, 0.8)
    return eng


def chambers_of(response):
    return {r.chamber for r in response.results}


def test_auto_routes_to_the_predicted_chamber(engine):
    full = engine.search("الطلاق النفقة", chamber="all")
    auto = engine.search("الطلاق النفقة", chamber="auto")
    assert chambers_of(auto) == {"personal_status"} and auto.chamber_routing["predicted"] == "personal_status"
    assert [s["chamber"] for s in auto.chamber_routing["searched"]] == ["personal_status"]
    assert auto.scored < full.scored or full.scored == auto.scored       # never more work than the full search
    assert [r.doc_id for r in auto.results] == [r.doc_id for r in full.results if r.chamber == "personal_status"]


def test_scores_are_the_same_with_or_without_routing(engine):
    full = {r.doc_id: r.score for r in engine.search("الشفعة", chamber="all").results}
    auto = {r.doc_id: r.score for r in engine.search("الشفعة", chamber="auto").results}
    assert auto and all(auto[d] == pytest.approx(full[d]) for d in auto)


def test_unknown_word_searches_every_chamber(engine):
    response = engine.search("الطلاق كلمةمجهولة", chamber="auto")
    routing = engine.search("كلمةمجهولة", chamber="auto").chamber_routing
    assert routing["reason"] == "no_known_word" and len(routing["searched"]) == 2
    assert response.results


def test_doubtful_query_searches_every_chamber(engine):
    original = engine.router.min_confidence
    engine.router.min_confidence = 0.999999
    try:
        routing = engine.search("الشفعة الطلاق", chamber="auto").chamber_routing
    finally:
        engine.router.min_confidence = original
    assert routing["reason"] == "low_confidence" and len(routing["searched"]) == 2


def test_manual_choice_all_and_errors(engine):
    assert chambers_of(engine.search("الارث", chamber="real_estate")) == set()
    assert engine.search("الارث", chamber="personal_status").results
    assert engine.search("النفقة", chamber="all").chamber_routing["mode"] == "all"
    assert engine.search("النفقة").chamber_routing["mode"] == "all"
    with pytest.raises(ValueError):
        engine.search("النفقة", chamber="criminal")


def test_a_category_filter_skips_the_routing(engine):
    engine.categories = {"1": "preemption", "4": "preemption"}
    try:
        response = engine.search("الشفعة", chamber="auto", category="preemption")
    finally:
        engine.categories = {}
    assert response.chamber_routing["reason"] == "category" and {r.doc_id for r in response.results} == {"1", "4"}


def test_without_a_router_auto_means_every_chamber(engine):
    router, engine.router = engine.router, None
    try:
        routing = engine.search("الطلاق", chamber="auto").chamber_routing
    finally:
        engine.router = router
    assert routing["reason"] == "no_model" and len(routing["searched"]) == 2


def test_one_chamber_only_has_no_routing_info():
    analyzer = analyzer_from_config(config, "light")
    docs = [d for d in DOCS if d["chamber"] == "real_estate"]
    eng = SearchEngine(analyzer, TfidfModel(InvertedIndex.build({str(d["doc_id"]): analyzer.analyze(d["body"]) for d in docs})), docs)
    assert eng.search("الشفعة", chamber="auto").chamber_routing is None


def test_similar_stays_in_the_chamber(engine):
    mixed = engine.similar("ps-1", k=7)
    same = engine.similar("ps-1", k=7, same_chamber=True)
    assert len(same) == 3 and all(d.startswith("ps-") for d, _ in same) and len(mixed) == 7


def test_partitions_of_one_chamber_are_used_only_inside_it(engine):
    analyzer = engine.analyzer
    terms = {str(d["doc_id"]): analyzer.analyze(d["body"]) for d in DOCS if d["chamber"] == "real_estate"}
    data = build_partitions(terms, 2, min_df=1, max_ratio=1.0) | {"chamber": "real_estate"}
    eng = SearchEngine(analyzer, engine.model, DOCS)
    eng.load_partitions(data)                                            # covers a subset of the index: accepted
    assert eng.partitions is not None and eng.partitions_chamber == "real_estate"
    assert eng.search("الشفعة", n_probe=1, chamber="real_estate").routing is not None
    assert eng.search("الشفعة", n_probe=1, chamber="all").routing is None
    assert eng.search("الارث", n_probe=1, chamber="personal_status").routing is None


def test_partitions_with_unknown_documents_are_refused(engine):
    eng = SearchEngine(engine.analyzer, engine.model, DOCS)
    with pytest.warns(UserWarning):
        eng.load_partitions({"assignments": {"zzz": 0}, "k": 1, "sizes": [1], "centroids": [{}], "idf": {}})
    assert eng.partitions is None


# ---------- API ----------
@pytest.fixture(scope="module")
def client(engine):
    return TestClient(create_app(engine))


def test_api_chamber_parameter_and_routing_info(client):
    data = client.get("/api/search", params={"q": "الطلاق النفقة", "chamber": "auto"}).json()
    assert {r["chamber"] for r in data["results"]} == {"personal_status"}
    assert data["chamber_routing"]["searched"] == [{"chamber": "personal_status", "name_ar": NAMES["personal_status"]}]
    assert data["chamber_routing"]["probabilities"][0]["chamber"] == "personal_status"
    assert client.get("/api/search", params={"q": "الطلاق", "chamber": "nope"}).status_code == 400


def test_api_stats_chambers_and_document(client):
    stats = client.get("/api/stats").json()
    assert stats["router"] is True and {c["key"]: c["count"] for c in stats["chambers"]} == {"real_estate": 4, "personal_status": 4}
    assert client.get("/api/chambers").json()["min_confidence"] == 0.8
    doc = client.get("/api/documents/ps-2").json()
    assert doc["chamber"] == "personal_status" and doc["chamber_ar"] == NAMES["personal_status"]


def test_api_similar_defaults_to_the_same_chamber(client):
    same = client.get("/api/documents/ps-1/similar", params={"k": 8}).json()["results"]
    assert len(same) == 3 and {r["chamber"] for r in same} == {"personal_status"}
    assert len(client.get("/api/documents/ps-1/similar", params={"k": 8, "same_chamber": "false"}).json()["results"]) == 7


# ---------- duplicates on a big collection ----------
def test_sparse_jaccard_matches_the_loops():
    import random
    rng = random.Random(1)
    vocab = [f"t{i}" for i in range(60)]
    terms = {str(i): rng.sample(vocab, 20) for i in range(40)}
    terms["40"] = list(terms["3"])                                       # an exact copy
    terms["41"] = terms["3"][:18] + ["x1", "x2"]
    slow = {(a, b): s for a, b, s in jaccard_pairs(terms, 0.3)}
    fast = {(a, b): s for a, b, s in _jaccard_pairs_sparse({k: set(v) for k, v in terms.items()}, 0.3)}
    assert fast.keys() == slow.keys() and all(fast[k] == pytest.approx(slow[k], rel=1e-5) for k in slow)
