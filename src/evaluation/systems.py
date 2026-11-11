"""The search systems that are compared. Each one has  rank(query, k) -> [doc_id, ...]  (best first).

  * ModeSystem   : the TF-IDF + cosine search of the project built with one morphology mode (word, light, root)
  * EngineSystem : the real SearchEngine (light mode), optionally with the Naive Bayes chamber routing
Decisions in `exclude` (the duplicates) are left out of every system, so the same text is never counted twice.
"""
from ..extraction.cleaner import strip_panel_names
from ..indexing.inverted_index import InvertedIndex
from ..indexing.tfidf import TfidfModel
from ..morphology.analyzer import analyzer_from_config


class ModeSystem:
    def __init__(self, config, documents: list[dict], mode: str, exclude: set | None = None):
        self.name = mode
        self.analyzer = analyzer_from_config(config, mode)
        keep = [d for d in documents if str(d["doc_id"]) not in (exclude or set())]
        terms = {str(d["doc_id"]): self.analyzer.analyze(strip_panel_names(d["body"])) for d in keep}
        self.model = TfidfModel(InvertedIndex.build(terms), getattr(config, "LOG_BASE", 10))

    def rank(self, query: str, k: int) -> list[str]:
        scores = self.model.cosine_scores(self.analyzer.analyze(query))
        ordered = sorted(((s, d) for d, s in scores.items() if s > 0), key=lambda sd: (-sd[0], sd[1]))
        return [d for _, d in ordered[:k]]


class EngineSystem:
    def __init__(self, engine, name: str, chamber: str | None = None, exclude: set | None = None):
        self.engine, self.name, self.chamber, self.exclude = engine, name, chamber, exclude or set()

    def rank(self, query: str, k: int) -> list[str]:
        response = self.engine.search(query, k=k + len(self.exclude), chamber=self.chamber)
        return [r.doc_id for r in response.results if r.doc_id not in self.exclude][:k]


def build_systems(config, documents: list[dict], exclude: set, engine=None) -> list:
    systems = [ModeSystem(config, documents, mode, exclude) for mode in ("word", "light", "root")]
    if engine is not None and getattr(engine, "router", None) is not None:
        systems.append(EngineSystem(engine, "light+routing", "auto", exclude))
    return systems
