"""Pipeline D: the search engine.

    query -> analyze() (the SAME morphology as the documents) -> look the terms up in the index
          -> candidate documents (those containing at least one term)
          -> TF-IDF vector of the query -> cosine with each candidate -> sort -> top k

Spelling: a query term that is not in the index gets "did you mean" suggestions (Levenshtein).
Boolean option: require_all=True keeps only the documents that contain ALL the query terms (AND).
"""
import json
import warnings
from dataclasses import dataclass, field

from ..indexing.inverted_index import InvertedIndex
from ..indexing.tfidf import TfidfModel
from ..morphology.analyzer import analyzer_from_config
from ..similarity import measures
from ..similarity.levenshtein import suggest
from .highlight import find_matches, make_snippet


@dataclass
class Result:
    rank: int
    doc_id: str
    score: float
    filename: str
    decision_no: str | None
    date: str | None
    case_no: str | None
    subject: str | None
    matched_terms: list[str]
    snippet: str


@dataclass
class SearchResponse:
    query: str
    terms: list[str]                                   # the analyzed query (stems)
    unknown: list[str] = field(default_factory=list)   # query terms absent from the index
    suggestions: dict = field(default_factory=dict)    # unknown term -> [(term, distance)]
    ignored: list[str] = field(default_factory=list)   # terms in every document (IDF = 0)
    total: int = 0                                     # documents with a positive score
    results: list[Result] = field(default_factory=list)


def _doc_order(doc_id: str):
    return (0, int(doc_id), "") if doc_id.isdigit() else (1, 0, doc_id)


class SearchEngine:
    def __init__(self, analyzer, model: TfidfModel, documents):
        self.analyzer = analyzer
        self.model = model
        self.index = model.index
        docs = documents.values() if isinstance(documents, dict) else documents
        self.documents = {str(d["doc_id"]): d for d in docs}
        self._vocabulary = None
        self._snippet_cache: dict = {}

    @classmethod
    def from_config(cls, config) -> "SearchEngine":
        analyzer = analyzer_from_config(config)
        index = InvertedIndex.load(config.INDEX_PKL)
        built_with = index.meta.get("mode")
        if built_with is None:
            warnings.warn("index.pkl has no build information: rebuild it with scripts/03_build_index.py")
        elif built_with != analyzer.mode:
            raise ValueError(f"index.pkl was built with mode {built_with!r} but config.ANALYZER_MODE is "
                             f"{analyzer.mode!r}: run scripts/02_analyze.py and 03_build_index.py again")
        model = TfidfModel(index, config.LOG_BASE)
        documents = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))
        return cls(analyzer, model, documents)

    # ---------- search ----------
    def search(self, query: str, k: int = 10, require_all: bool = False) -> SearchResponse:
        terms = self.analyzer.analyze(query)
        response = SearchResponse(query=query, terms=terms)
        if not terms:
            return response

        known = sorted({t for t in terms if t in self.index.postings})
        response.unknown = sorted({t for t in terms if t not in self.index.postings})
        response.ignored = [t for t in known if self.model.idf[t] == 0]
        for term in response.unknown:
            response.suggestions[term] = suggest(term, self._vocab(), k=3, df=self.index.df)

        scores = self.model.cosine_scores(terms)
        if require_all:                                  # Boolean AND on the known terms
            if response.unknown:
                scores = {}                              # an unknown term can never be in a document
            else:
                keep = set.intersection(*(set(self.index.postings[t]) for t in known)) if known else set()
                scores = {d: s for d, s in scores.items() if d in keep}
        scores = {d: s for d, s in scores.items() if s > 0}

        response.total = len(scores)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], _doc_order(item[0])))[:k]
        response.results = [self._result(rank, doc_id, score, terms)
                            for rank, (doc_id, score) in enumerate(ranked, 1)]
        return response

    def _result(self, rank: int, doc_id: str, score: float, terms: list[str]) -> Result:
        doc = self.documents.get(doc_id, {})
        body = doc.get("body", "")
        spans = find_matches(body, terms, self.analyzer, self._snippet_cache)
        return Result(
            rank=rank, doc_id=doc_id, score=score, filename=doc.get("filename", f"{doc_id}.txt"),
            decision_no=doc.get("decision_no"), date=doc.get("date"), case_no=doc.get("case_no"),
            subject=doc.get("subject"),
            matched_terms=sorted(t for t in set(terms) if self.index.count(t, doc_id) > 0),
            snippet=make_snippet(body, spans),
        )

    # ---------- documents similar to a given document ----------
    def similar(self, doc_id, k: int = 10, measure: str = "cosine") -> list[tuple[str, float]]:
        """The k documents closest to `doc_id`. For distances (euclidean, manhattan) smaller is closer."""
        doc_id = str(doc_id)
        if doc_id not in self.index.doc_len:
            raise KeyError(f"unknown document {doc_id!r}")
        if measure not in measures.MEASURES:
            raise ValueError(f"measure must be one of {sorted(measures.MEASURES)}")
        function, works_on, is_distance = measures.MEASURES[measure]

        def representation(d):
            vector = self.model.doc_vector(d)
            return vector if works_on == "vectors" else set(vector)

        base = representation(doc_id)
        scored = [(d, function(base, representation(d))) for d in self.index.doc_len if d != doc_id]
        scored.sort(key=lambda item: (item[1] if is_distance else -item[1], _doc_order(item[0])))
        return scored[:k]

    def _vocab(self):
        if self._vocabulary is None:
            self._vocabulary = self.index.vocabulary
        return self._vocabulary
