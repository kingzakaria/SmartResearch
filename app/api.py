"""HTTP API of the search engine (FastAPI). The React interface talks to it.

Run from the project root:   python -m uvicorn app.api:app --reload --port 8000
Interactive documentation:   http://localhost:8000/docs
"""
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

import config  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402
from src.search.highlight import find_matches, reading_text, segment_text  # noqa: E402
from src.similarity import measures  # noqa: E402

MAX_K = 50


def create_app(engine: SearchEngine | None = None) -> FastAPI:
    """`engine` can be injected (tests); otherwise it is loaded from models/index.pkl on first use."""
    app = FastAPI(title="Arabic court decisions search", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET"], allow_headers=["*"])
    state = {"engine": engine}

    def get_engine() -> SearchEngine:
        if state["engine"] is None:
            try:
                state["engine"] = SearchEngine.from_config(config)
            except FileNotFoundError as error:
                raise HTTPException(503, "Index not found. Run: python scripts/01_extract.py, "
                                         "python scripts/02_analyze.py, python scripts/03_build_index.py "
                                         f"({error.filename})")
            except ValueError as error:
                raise HTTPException(503, str(error))
        return state["engine"]

    @app.get("/api/health")
    def health():
        engine = get_engine()
        return {"status": "ok", "documents": engine.index.n_docs}

    @app.get("/api/stats")
    def stats():
        engine = get_engine()
        years = sorted({d["year"] for d in engine.documents.values() if d.get("year")})
        return {"documents": engine.index.n_docs, "terms": len(engine.index.postings),
                "mode": engine.analyzer.mode, "years": years,
                "measures": sorted(measures.MEASURES)}

    @app.get("/api/search")
    def search(q: str = Query("", max_length=500), k: int = Query(10, ge=1, le=MAX_K), all: bool = False):
        response = get_engine().search(q, k=k, require_all=all)
        data = asdict(response)
        data["suggestions"] = {t: [{"term": s, "distance": d} for s, d in found]
                               for t, found in response.suggestions.items()}
        return data

    @app.get("/api/documents/{doc_id}")
    def document(doc_id: str, q: str = Query("", max_length=500)):
        engine = get_engine()
        doc = engine.documents.get(doc_id)
        if doc is None:
            raise HTTPException(404, f"Document {doc_id} not found")
        terms = engine.analyzer.analyze(q) if q else []
        text = reading_text(doc.get("body", ""), doc.get("subject"))
        spans = find_matches(text, terms, engine.analyzer) if terms else []
        subject = doc.get("subject") or ""
        subject_spans = find_matches(subject, terms, engine.analyzer) if terms else []
        return {
            "subject_segments": segment_text(subject, subject_spans),
            "doc_id": doc_id, "filename": doc.get("filename"), "decision_no": doc.get("decision_no"),
            "date": doc.get("date"), "case_no": doc.get("case_no"), "subject": doc.get("subject"),
            "n_words": doc.get("n_words"), "terms": terms, "matches": len(spans) + len(subject_spans),
            "body": segment_text(text, spans), "closing": doc.get("closing") or "",
        }

    @app.get("/api/documents/{doc_id}/similar")
    def similar(doc_id: str, k: int = Query(8, ge=1, le=MAX_K), measure: str = "cosine"):
        engine = get_engine()
        if measure not in measures.MEASURES:
            raise HTTPException(400, f"measure must be one of {sorted(measures.MEASURES)}")
        try:
            found = engine.similar(doc_id, k=k, measure=measure)
        except KeyError:
            raise HTTPException(404, f"Document {doc_id} not found")
        is_distance = measures.MEASURES[measure][2]
        items = []
        for other, value in found:
            d = engine.documents.get(other, {})
            items.append({"doc_id": other, "value": value, "decision_no": d.get("decision_no"),
                          "date": d.get("date"), "case_no": d.get("case_no"), "subject": d.get("subject")})
        return {"doc_id": doc_id, "measure": measure, "is_distance": is_distance, "results": items}

    return app


app = create_app()
