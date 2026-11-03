"""HTTP API of the search engine (FastAPI). The React interface talks to it.

Run from the project root:   python -m uvicorn app.api:app --reload --port 8000
Interactive documentation:   http://localhost:8000/docs
"""
import sys
import threading
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402
from src.search.highlight import find_matches, reading_text, segment_text  # noqa: E402
from src.similarity import measures  # noqa: E402

MAX_K = 50


class ClassifyRequest(BaseModel):
    text: str = Field(max_length=20000)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


def create_app(engine: SearchEngine | None = None, classifier=None, rag=None) -> FastAPI:
    """`engine` can be injected (tests); otherwise it is loaded from models/index.pkl on first use."""
    app = FastAPI(title="Arabic court decisions search", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    state = {"engine": engine, "classifier": classifier, "classifier_tried": classifier is not None, "rag": rag}
    lock = threading.Lock()           # the first requests arrive together: load each model only once, and wait for it

    def get_engine() -> SearchEngine:
        with lock:
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

    def get_classifier():
        """Loaded on first use; None when models/classifier.pkl does not exist (or scikit-learn is not installed)."""
        analyzer = get_engine().analyzer
        with lock:
            if not state["classifier_tried"]:
                try:
                    from src.classification.service import Classifier
                    state["classifier"] = Classifier.from_config(config, analyzer)
                except (FileNotFoundError, ImportError):
                    state["classifier"] = None
                except ValueError as error:
                    raise HTTPException(503, str(error))
                state["classifier_tried"] = True          # only after the loading is finished
            return state["classifier"]

    def get_rag():
        """The "Ask AI" service, loaded on first use. 503 with the command to run when something is missing."""
        engine = get_engine()
        with lock:
            if state["rag"] is None:
                from src.rag.store import build_service
                try:
                    state["rag"] = build_service(config, engine)
                except FileNotFoundError:
                    raise HTTPException(503, "No passages yet. Run: python scripts/12_build_chunks.py "
                                             "then python scripts/13_embed_chunks.py")
                except PermissionError:
                    raise HTTPException(503, "No Gemini API key. Create the file .env at the project root "
                                             "containing: GEMINI_API_KEY=your_key_here  then restart the API")
                except ValueError as error:
                    raise HTTPException(503, str(error))
            return state["rag"]

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
                "partitions": None if engine.partitions is None else
                {"k": engine.partitions.k, "sizes": engine.partitions.sizes},
                "classifier": get_classifier() is not None,
                "rag": rag_status(),
                "measures": sorted(measures.MEASURES)}

    def rag_status() -> dict:
        if state["rag"] is not None:
            return {"ready": True, "hybrid": state["rag"].retriever.hybrid}
        from src.rag.store import rag_files_status
        files = rag_files_status(config)
        return {"ready": files["chunks"] and files["key"], "hybrid": files["chunks"] and files["embeddings"] and files["key"]}

    @app.post("/api/ask")
    def ask(request: AskRequest):
        """Answer a question from the decisions: hybrid retrieval of passages, then Gemini writes the answer
        and cites the passages it used."""
        from src.rag.gemini import GeminiError
        service = get_rag()
        try:
            return service.ask(request.question.strip())
        except GeminiError as error:
            if error.status == 429:
                raise HTTPException(429, "Gemini free-tier limit reached. Wait a minute and ask again.")
            raise HTTPException(502, f"Gemini error: {error}")

    @app.get("/api/categories")
    def categories():
        """The categories (from the subject lines) with their Arabic name and number of decisions."""
        engine = get_engine()
        counts: dict[str, int] = {}
        for category in engine.categories.values():
            counts[category] = counts.get(category, 0) + 1
        return {"categories": [{"name": name, "name_ar": engine.category_names.get(name, name), "count": count}
                               for name, count in sorted(counts.items(), key=lambda kv: -kv[1])]}

    @app.post("/api/classify")
    def classify(request: ClassifyRequest):
        """Predict the category of a pasted text, with a probability for every category."""
        classifier = get_classifier()
        if classifier is None:
            raise HTTPException(503, "No classifier yet. Run: python scripts/10_label_by_subject.py, "
                                     "then python scripts/11_train_classifiers.py")
        engine = get_engine()
        predictions = classifier.predict(request.text)
        return {"predictions": [{"name": c, "name_ar": engine.category_names.get(c, c), "probability": round(p, 4)}
                                for c, p in predictions]}

    @app.get("/api/partitions")
    def partitions():
        """The groups the collection was divided in: size and main terms of each."""
        p = get_engine().partitions
        if p is None:
            return {"k": 0, "groups": []}
        return {"k": p.k, "groups": [{"group": i, "size": p.sizes[i], "top_terms": p.top_terms[i][:10]}
                                     for i in range(p.k)]}

    @app.get("/api/search")
    def search(q: str = Query("", max_length=500), k: int = Query(10, ge=1, le=MAX_K), all: bool = False,
               n_probe: int = Query(0, ge=0, le=20, description="0 = search everything; n = only the n best groups"),
               category: str = Query("", max_length=60)):
        response = get_engine().search(q, k=k, require_all=all, n_probe=n_probe or None,
                                           category=category or None)
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
            "category": engine.categories.get(doc_id),
            "category_ar": engine.category_names.get(engine.categories.get(doc_id) or "", None),
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
