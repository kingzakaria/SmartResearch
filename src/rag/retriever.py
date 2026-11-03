"""Step 2 of RAG: find the passages that may answer a question (hybrid retrieval).

Two retrievers look at the same passages and each proposes its best ones:
  * keywords:   the TF-IDF + cosine engine of the course, over passages, with the same morphology as the search page.
                Strong on exact legal terms and article numbers; blind to synonyms.
  * embeddings: Gemini vectors. A question and a passage that say the same thing in different words are close.
                Strong on meaning; weaker on rare exact terms.
Their two rankings are merged with Reciprocal Rank Fusion:  score(p) = sum over retrievers of 1 / (k + rank(p)).
RRF only uses ranks, so the two incompatible score scales (cosine of sparse vectors / cosine of dense vectors)
never have to be compared.

Without embeddings (not computed yet, or Gemini unreachable) the retriever works with keywords only.
"""
from dataclasses import dataclass

import numpy as np

from ..indexing.inverted_index import InvertedIndex
from ..indexing.tfidf import TfidfModel


@dataclass
class Passage:
    n: int                      # 1, 2, 3 ...: the number the LLM cites as [n]
    chunk_id: str
    doc_id: str
    text: str
    score: float                # fused RRF score
    keyword_rank: int | None    # rank in the keyword list (None: not proposed by it)
    embedding_rank: int | None


def fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion of several ranked lists of ids."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, analyzer, chunks: list[dict], embeddings: tuple | None = None, embedder=None, log_base: float = 10,
                 candidates: int = 20, rrf_k: int = 60, max_per_doc: int = 2):
        """chunks: [{chunk_id, doc_id, text, terms}]; embeddings: (ids, matrix) with unit-length rows."""
        self.analyzer = analyzer
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.index = InvertedIndex.build({c["chunk_id"]: c["terms"] for c in chunks})
        self.model = TfidfModel(self.index, log_base)
        self.candidates, self.rrf_k, self.max_per_doc = candidates, rrf_k, max_per_doc
        self.embedder = embedder
        self.ids = self.matrix = None
        if embeddings is not None:
            ids, matrix = embeddings
            if set(ids) == set(self.chunks):                  # the vectors must belong to these exact passages
                self.ids, self.matrix = list(ids), matrix
        self.last_warning: str | None = None

    @property
    def hybrid(self) -> bool:
        return self.matrix is not None and self.embedder is not None

    def _keyword_ranking(self, question: str) -> list[str]:
        scores = {c: s for c, s in self.model.cosine_scores(self.analyzer.analyze(question)).items() if s > 0}
        return [c for c, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:self.candidates]]

    def _embedding_ranking(self, question: str) -> list[str]:
        similarity = self.matrix @ self.embedder.embed_query(question)
        order = np.argsort(-similarity)[:self.candidates]
        return [self.ids[i] for i in order]

    def retrieve(self, question: str, top: int = 5) -> list[Passage]:
        self.last_warning = None
        keyword = self._keyword_ranking(question)
        embedding: list[str] = []
        if self.hybrid:
            try:
                embedding = self._embedding_ranking(question)
            except Exception as error:                      # Gemini down or over quota: keep working with keywords
                self.last_warning = f"Embeddings unavailable ({error}); keyword retrieval only."
        fused = fuse([r for r in (keyword, embedding) if r], self.rrf_k)

        kw_rank = {c: i for i, c in enumerate(keyword, 1)}
        em_rank = {c: i for i, c in enumerate(embedding, 1)}
        chosen, per_doc = [], {}
        for chunk_id, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])):
            doc_id = self.chunks[chunk_id]["doc_id"]
            if per_doc.get(doc_id, 0) >= self.max_per_doc:
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            chosen.append(Passage(len(chosen) + 1, chunk_id, doc_id, self.chunks[chunk_id]["text"], score,
                                  kw_rank.get(chunk_id), em_rank.get(chunk_id)))
            if len(chosen) == top:
                break
        return chosen
