"""HTTP API on the small in-memory corpus of test_search. Run: pytest tests/test_api.py"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

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
