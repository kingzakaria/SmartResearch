"""Pipeline D: the search engine.

    query -> analyze() (the SAME morphology as the documents) -> look the terms up in the index
          -> candidate documents (those containing at least one term)
          -> TF-IDF vector of the query -> cosine with each candidate -> sort -> top k

Spelling: a query term that is not in the index gets "did you mean" suggestions (Levenshtein).
Boolean option: require_all=True keeps only the documents that contain ALL the query terms (AND).
"""
import json
import time
import warnings
from dataclasses import dataclass, field

from ..indexing.inverted_index import InvertedIndex
from ..indexing.tfidf import TfidfModel
from ..morphology.analyzer import analyzer_from_config
from ..similarity import measures
from ..similarity.levenshtein import suggest
from .highlight import find_matches, make_snippet
from .chamber_router import ChamberRouter
from .partitions import Partitions

DEFAULT_CHAMBER = "real_estate"        # documents written before the second chamber have no "chamber" field


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
    category: str | None = None                        # category of the decision (from its subject line)
    group: int | None = None                           # partition of the document (None without partitions)
    chamber: str | None = None                         # the chamber (class) the decision belongs to


@dataclass
class SearchResponse:
    query: str
    terms: list[str]                                   # the analyzed query (stems)
    unknown: list[str] = field(default_factory=list)   # query terms absent from the index
    suggestions: dict = field(default_factory=dict)    # unknown term -> [(term, distance)]
    ignored: list[str] = field(default_factory=list)   # terms in every document (IDF = 0)
    total: int = 0                                     # documents with a positive score
    results: list[Result] = field(default_factory=list)
    scored: int = 0                                    # documents whose score was computed
    took_ms: float = 0.0
    routing: dict | None = None                        # partitioned search: groups searched, similarity to centroids
    chamber_routing: dict | None = None                # which chamber(s) were searched and why (Naive Bayes on the query)


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
        self.partitions: Partitions | None = None
        self.categories: dict[str, str] = {}               # doc_id -> category
        self.category_names: dict[str, str] = {}           # category -> Arabic name
        self.chambers: dict[str, str] = {d: doc.get("chamber") or DEFAULT_CHAMBER for d, doc in self.documents.items()}
        self.chamber_names: dict[str, str] = {}            # chamber -> Arabic name
        self.router: ChamberRouter | None = None           # Naive Bayes on the query (None until scripts/15 has run)
        self.partitions_chamber: str | None = None         # the chamber the K-Means groups were built for

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
        engine = cls(analyzer, model, documents)
        engine._load_categories(config)
        engine.chamber_names = {k: v["name_ar"] for k, v in getattr(config, "CHAMBERS", {}).items()}
        engine._load_router(config)
        path = getattr(config, "PARTITIONS_JSON", None)
        if path and path.exists():
            engine.load_partitions(json.loads(path.read_text(encoding="utf-8")))
        return engine

    def _load_router(self, config) -> None:
        """The chamber router is optional: without models/naive_bayes.json the search simply covers every chamber."""
        from ..classification.naive_bayes import NaiveBayes
        path = getattr(config, "NB_MODEL_JSON", None)
        if not (path and path.exists()):
            return
        model, meta = NaiveBayes.load(path)
        if meta.get("mode") not in (None, self.analyzer.mode):
            warnings.warn(f"naive_bayes.json was built with mode {meta.get('mode')!r}: run scripts/15_train_naive_bayes.py again")
            return
        self.router = ChamberRouter(model, getattr(config, "CHAMBER_MIN_CONFIDENCE", 0.8),
                                    getattr(config, "NB_PRIORS", "empirical") == "uniform")

    def _load_categories(self, config) -> None:
        """Categories are optional: without labels.csv the search simply has no category filter."""
        from ..analysis.categories import category_names, propagate_to_duplicates, read_labels
        labels_path = getattr(config, "LABELS_CSV", None)
        if not (labels_path and labels_path.exists()):
            return
        dup_path = getattr(config, "DUPLICATES_JSON", None)
        dup = json.loads(dup_path.read_text(encoding="utf-8")) if dup_path and dup_path.exists() else {}
        self.categories = propagate_to_duplicates(read_labels(labels_path), dup.get("groups", []), dup.get("drop", []))
        rules_path = config.RESOURCES_DIR / "category_rules.json"
        self.category_names = category_names(rules_path) if rules_path.exists() else {}

    def load_partitions(self, data: dict) -> None:
        """Enable the partitioned search. Ignored (with a warning) if it was built for other documents."""
        members = set(map(str, data["assignments"]))
        if not members or not members <= set(self.index.doc_len):
            warnings.warn("partitions.json does not match the index: run scripts/07_build_partitions.py again")
            return
        self.partitions = Partitions(data, self.index, self.model)
        chambers = {self.chambers.get(d) for d in members}
        self.partitions_chamber = data.get("chamber") or (chambers.pop() if len(chambers) == 1 else None)

    # ---------- search ----------
    def search(self, query: str, k: int = 10, require_all: bool = False, n_probe: int | None = None,
               category: str | None = None, chamber: str | None = None) -> SearchResponse:
        """n_probe=None searches every document; n_probe=1 only the best group, 2 the best two groups, ...

        chamber: None or "all" = every chamber; "auto" = Naive Bayes predicts the chamber of the query and only that
        one is searched (every chamber when the prediction is not sure); a chamber key = that chamber only."""
        started = time.perf_counter()
        terms = self.analyzer.analyze(query)
        response = SearchResponse(query=query, terms=terms)
        if not terms:
            return response

        known = sorted({t for t in terms if t in self.index.postings})
        response.unknown = sorted({t for t in terms if t not in self.index.postings})
        response.ignored = [t for t in known if self.model.idf[t] == 0]
        for term in response.unknown:
            response.suggestions[term] = suggest(term, self._vocab(), k=3, df=self.index.df)

        chambers, response.chamber_routing = self._choose_chambers(terms, chamber, category)
        everywhere = set(self.chambers.values())
        scope = set(chambers) if chambers else everywhere
        allowed = None if scope >= everywhere else {d for d, c in self.chambers.items() if c in scope}

        # the K-Means groups cover one chamber: they are used only when the search is inside that chamber
        if n_probe and self.partitions is not None and scope == {self.partitions_chamber}:
            groups, sims, fallback = self.partitions.route(terms, n_probe)
            scores = self.partitions.cosine_scores(terms, groups)
            response.routing = {"groups": groups, "similarity": [round(s, 4) for s in sims],
                                "fallback": fallback, "k": self.partitions.k}
        else:
            scores = self.model.cosine_scores(terms, allowed)
        response.scored = len(scores)
        if require_all:                                  # Boolean AND on the known terms
            if response.unknown:
                scores = {}                              # an unknown term can never be in a document
            else:
                keep = set.intersection(*(set(self.index.postings[t]) for t in known)) if known else set()
                scores = {d: s for d, s in scores.items() if d in keep}
        scores = {d: s for d, s in scores.items() if s > 0}
        if category:                                     # keep only the decisions of this category
            scores = {d: s for d, s in scores.items() if self.categories.get(d) == category}

        response.total = len(scores)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], _doc_order(item[0])))[:k]
        response.results = [self._result(rank, doc_id, score, terms)
                            for rank, (doc_id, score) in enumerate(ranked, 1)]
        response.took_ms = round((time.perf_counter() - started) * 1000, 2)
        return response

    def _choose_chambers(self, terms: list[str], chamber: str | None, category: str | None):
        """(chambers to search or None for all, explanation shown to the user)."""
        known = sorted(set(self.chambers.values()))
        if len(known) < 2:
            return None, None                                   # one chamber only: nothing to choose
        names = lambda keys: [{"chamber": k, "name_ar": self.chamber_names.get(k, k)} for k in keys]  # noqa: E731
        if chamber in (None, "", "all"):
            return None, {"mode": "all", "predicted": None, "probabilities": [], "searched": names(known), "reason": "chosen"}
        if chamber != "auto":
            if chamber not in known:
                raise ValueError(f"unknown chamber {chamber!r}; choose one of {known} or 'auto' or 'all'")
            return [chamber], {"mode": "manual", "predicted": None, "probabilities": [], "searched": names([chamber]),
                               "reason": "chosen"}
        if category:                                            # categories exist for one chamber: the filter decides
            return None, {"mode": "auto", "predicted": None, "probabilities": [], "searched": names(known),
                          "reason": "category"}
        if self.router is None:
            return None, {"mode": "auto", "predicted": None, "probabilities": [], "searched": names(known),
                          "reason": "no_model"}
        decision = self.router.route(terms)
        probabilities = [{"chamber": c, "name_ar": self.chamber_names.get(c, c), "probability": round(p, 4)}
                         for c, p in sorted(decision.probabilities.items(), key=lambda kv: -kv[1])]
        return decision.searched, {"mode": "auto", "predicted": decision.predicted, "probabilities": probabilities,
                                   "searched": names(decision.searched), "reason": decision.reason}

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
            category=self.categories.get(doc_id),
            group=self.partitions.group_of(doc_id) if self.partitions else None,
            chamber=self.chambers.get(doc_id),
        )

    # ---------- documents similar to a given document ----------
    def similar(self, doc_id, k: int = 10, measure: str = "cosine", same_chamber: bool = False) -> list[tuple[str, float]]:
        """The k documents closest to `doc_id`. For distances (euclidean, manhattan) smaller is closer.
        same_chamber=True only looks at the decisions of the same chamber as `doc_id`."""
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
        mine = self.chambers.get(doc_id)
        scored = [(d, function(base, representation(d))) for d in self.index.doc_len
                  if d != doc_id and (not same_chamber or self.chambers.get(d) == mine)]
        scored.sort(key=lambda item: (item[1] if is_distance else -item[1], _doc_order(item[0])))
        return scored[:k]

    def _vocab(self):
        if self._vocabulary is None:
            self._vocabulary = self.index.vocabulary
        return self._vocabulary
