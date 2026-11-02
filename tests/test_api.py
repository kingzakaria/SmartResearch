"""HTTP API on the small in-memory corpus of test_search. Run: pytest tests/test_api.py"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from app.api import create_app  # noqa: E402
from src.indexing.inverted_index import InvertedIndex  # noqa: E402
from src.indexing.tfidf import TfidfModel  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402
from test_search import DOCS  # noqa: E402


@pytest.fixture(scope="module")
def client():
    analyzer = analyzer_from_config(config, "light")
    index = InvertedIndex.build({d["doc_id"]: analyzer.analyze(d["body"]) for d in DOCS})
    return TestClient(create_app(SearchEngine(analyzer, TfidfModel(index), DOCS)))


def test_health_and_stats(client):
    assert client.get("/api/health").json() == {"status": "ok", "documents": 6}
    stats = client.get("/api/stats").json()
    assert stats["documents"] == 6 and "cosine" in stats["measures"]


def test_search(client):
    data = client.get("/api/search", params={"q": "الشفعة"}).json()
    assert [r["doc_id"] for r in data["results"]] and data["total"] == 2
    assert "«الشفعة»" in data["results"][0]["snippet"] or "«بالشفعة»" in data["results"][0]["snippet"]


def test_search_suggestions_are_objects(client):
    data = client.get("/api/search", params={"q": "شفعا"}).json()
    assert data["suggestions"]["شفعا"][0]["term"] == "شفع"


def test_search_and_mode_and_k_validation(client):
    assert client.get("/api/search", params={"q": "الشفعة العقار", "all": "true"}).json()["total"] == 2
    assert client.get("/api/search", params={"q": "x", "k": 0}).status_code == 422
    assert client.get("/api/search", params={"q": ""}).json()["results"] == []


def test_document_with_highlighted_segments(client):
    data = client.get("/api/documents/1", params={"q": "الشفعة"}).json()
    assert data["matches"] == 2
    assert "".join(s["t"] for s in data["body"]) == DOCS[0]["body"]
    assert {s["t"] for s in data["body"] if s["m"]} == {"الشفعة", "بالشفعة"}


def test_document_subject_is_highlighted_too(client):
    data = client.get("/api/documents/1", params={"q": "موضوع"}).json()
    assert data["subject_segments"] == [{"t": "موضوع", "m": True}, {"t": " 1", "m": False}] and data["matches"] == 1


def test_document_without_query_and_unknown_document(client):
    assert client.get("/api/documents/1").json()["matches"] == 0
    assert client.get("/api/documents/999").status_code == 404


def test_similar(client):
    data = client.get("/api/documents/1/similar", params={"k": 1}).json()
    assert data["results"][0]["doc_id"] == "4" and not data["is_distance"]
    assert client.get("/api/documents/1/similar", params={"measure": "euclidean"}).json()["is_distance"]
    assert client.get("/api/documents/1/similar", params={"measure": "x"}).status_code == 400
    assert client.get("/api/documents/999/similar").status_code == 404


def test_missing_index_gives_an_instructive_503(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_PKL", tmp_path / "nope.pkl")
    response = TestClient(create_app()).get("/api/health")
    assert response.status_code == 503 and "03_build_index" in response.json()["detail"]


def test_partitioned_search_and_group_listing():
    from src.clustering.partitioning import build_partitions
    analyzer = analyzer_from_config(config, "light")
    terms = {d["doc_id"]: analyzer.analyze(d["body"]) for d in DOCS}
    engine = SearchEngine(analyzer, TfidfModel(InvertedIndex.build(terms)), DOCS)
    engine.load_partitions(build_partitions({str(d): t for d, t in terms.items()}, 2, min_df=1, max_ratio=1.0))
    client = TestClient(create_app(engine))
    assert client.get("/api/stats").json()["partitions"]["k"] == 2
    groups = client.get("/api/partitions").json()
    assert groups["k"] == 2 and sum(g["size"] for g in groups["groups"]) == 6
    full = client.get("/api/search", params={"q": "الشفعة"}).json()
    part = client.get("/api/search", params={"q": "الشفعة", "n_probe": 1}).json()
    assert full["routing"] is None and part["routing"]["groups"] and part["scored"] <= full["scored"]
    assert all(r["group"] is not None for r in part["results"])


def test_without_partitions_the_group_list_is_empty(client):
    assert client.get("/api/partitions").json() == {"k": 0, "groups": []}
    assert client.get("/api/search", params={"q": "الشفعة", "n_probe": 1}).json()["routing"] is None


# ---------- categories and classifier ----------
def make_engine_with_categories():
    analyzer = analyzer_from_config(config, "light")
    terms = {d["doc_id"]: analyzer.analyze(d["body"]) for d in DOCS}
    engine = SearchEngine(analyzer, TfidfModel(InvertedIndex.build(terms)), DOCS)
    engine.categories = {"1": "preemption_partition", "4": "preemption_partition", "2": "registration"}
    engine.category_names = {"preemption_partition": "الشفعة", "registration": "التحفيظ"}
    return engine


def test_categories_filter_and_listing():
    client = TestClient(create_app(make_engine_with_categories()))
    listing = client.get("/api/categories").json()["categories"]
    assert listing[0] == {"name": "preemption_partition", "name_ar": "الشفعة", "count": 2}
    both = client.get("/api/search", params={"q": "الشفعة العقار"}).json()
    assert {r["doc_id"]: r["category"] for r in both["results"]}["2"] == "registration"
    only = client.get("/api/search", params={"q": "الشفعة العقار", "category": "registration"}).json()
    assert [r["doc_id"] for r in only["results"]] == ["2"] and only["total"] == 1
    assert client.get("/api/documents/1").json()["category_ar"] == "الشفعة"


def test_classify_without_a_model_explains_what_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CLASSIFIER_PKL", tmp_path / "no_such_classifier.pkl")   # whatever exists on this machine
    response = TestClient(create_app(make_engine_with_categories())).post("/api/classify", json={"text": "شفعة"})
    assert response.status_code == 503 and "11_train_classifiers" in response.json()["detail"]


def test_classify_returns_probabilities():
    pytest.importorskip("sklearn")
    from src.classification.evaluate import make_pipeline
    from src.classification.service import Classifier
    engine = make_engine_with_categories()
    tokens = [["شفع", "بيع", "عقار"]] * 6 + [["تحفيظ", "تعرض", "مطلب"]] * 6
    pipe = make_pipeline("logistic_regression", 100, 1, 1.0).fit(tokens, ["a"] * 6 + ["b"] * 6)
    client = TestClient(create_app(engine, classifier=Classifier(pipe, ["a", "b"], engine.analyzer)))
    body = client.post("/api/classify", json={"text": "نزاع حول الشفعة وبيع العقار"}).json()["predictions"]
    assert body[0]["name"] == "a" and abs(sum(p["probability"] for p in body) - 1) < 1e-3
    assert client.post("/api/classify", json={"text": "في من على"}).json()["predictions"] == []
    assert client.get("/api/stats").json()["classifier"] is True


def test_concurrent_first_requests_all_see_the_classifier():
    pytest.importorskip("sklearn")
    import threading
    from src.classification.evaluate import make_pipeline
    from src.classification.service import Classifier
    from types import SimpleNamespace
    import pickle
    import tempfile
    engine = make_engine_with_categories()
    tokens = [["شفع", "بيع", "عقار"]] * 6 + [["تحفيظ", "تعرض", "مطلب"]] * 6
    pipe = make_pipeline("logistic_regression", 100, 1, 1.0).fit(tokens, ["a"] * 6 + ["b"] * 6)
    path = Path(tempfile.mkdtemp()) / "classifier.pkl"
    path.write_bytes(pickle.dumps({"pipeline": pipe, "classes": ["a", "b"], "mode": "light"}))
    original = config.CLASSIFIER_PKL
    config.CLASSIFIER_PKL = path
    try:
        client = TestClient(create_app(engine))
        codes = []
        threads = [threading.Thread(target=lambda: codes.append(
            client.post("/api/classify", json={"text": "الشفعة وبيع العقار"}).status_code)) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert codes == [200] * 6
    finally:
        config.CLASSIFIER_PKL = original
