"""TF-IDF weighting and cosine similarity, written by hand with the formulas of the course.

    TF(i,j)  = w(i,j) / |d_j|                 occurrences of term i in document j / terms of document j
    IDF(i)   = log10( |D| / df(i) )           |D| = number of documents, df = documents containing i
    weight   = TF(i,j) x IDF(i)
    cos(q,d) = (q . d) / (||q|| x ||d||)

The course uses log base 10 (its example 0.0092 only works with log10). scikit-learn's
TfidfVectorizer uses other conventions (natural log, smoothing), so its numbers differ.

Vectors are SPARSE dictionaries {term: weight}: a document holds ~600 terms out of a vocabulary
of ~10,000, so storing zeros would waste memory and time.
"""
import math

from .inverted_index import InvertedIndex


def tf(count: int, doc_len: int) -> float:
    return count / doc_len if doc_len else 0.0


def idf(n_docs: int, df: int, base: float = 10) -> float:
    """0 for a term that appears in every document (it cannot discriminate) or in none."""
    return math.log(n_docs / df, base) if df else 0.0


def weight(count: int, doc_len: int, n_docs: int, df: int, base: float = 10) -> float:
    return tf(count, doc_len) * idf(n_docs, df, base)


def _norm(vector: dict) -> float:
    return math.sqrt(sum(w * w for w in vector.values()))


class TfidfModel:
    def __init__(self, index: InvertedIndex, log_base: float = 10):
        self.index = index
        self.log_base = log_base
        n = index.n_docs
        self.idf = {term: idf(n, len(docs), log_base) for term, docs in index.postings.items()}

        self.vectors: dict[str, dict[str, float]] = {d: {} for d in index.doc_len}
        for term, docs in index.postings.items():
            term_idf = self.idf[term]
            for doc_id, count in docs.items():
                self.vectors[doc_id][term] = tf(count, index.doc_len[doc_id]) * term_idf
        self.norms = {d: _norm(v) for d, v in self.vectors.items()}

    # ---------- vectors ----------
    def doc_vector(self, doc_id) -> dict[str, float]:
        return self.vectors[str(doc_id)]

    def query_vector(self, terms: list[str]) -> dict[str, float]:
        """Same formula as for documents, with the query as a tiny document.
        Terms unknown to the index are ignored (they cannot match anything)."""
        if not terms:
            return {}
        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        return {t: tf(c, len(terms)) * self.idf[t] for t, c in counts.items() if t in self.idf}

    def top_terms(self, doc_id, k: int = 10) -> list[tuple[str, float]]:
        """The k heaviest terms of a document: its automatic keywords."""
        vector = self.doc_vector(doc_id)
        return sorted(vector.items(), key=lambda kv: kv[1], reverse=True)[:k]

    # ---------- similarity ----------
    def cosine(self, query_vec: dict[str, float], doc_id) -> float:
        doc_id = str(doc_id)
        q_norm, d_norm = _norm(query_vec), self.norms[doc_id]
        if not q_norm or not d_norm:
            return 0.0
        doc_vec = self.vectors[doc_id]
        dot = sum(w * doc_vec[t] for t, w in query_vec.items() if t in doc_vec)   # only shared terms
        return dot / (q_norm * d_norm)

    def cosine_scores(self, terms: list[str]) -> dict[str, float]:
        """Cosine of the query against every CANDIDATE document (those sharing at least one term).
        This is where the inverted index pays off: other documents are never touched."""
        query_vec = self.query_vector(terms)
        return {d: self.cosine(query_vec, d) for d in self.index.candidates(list(query_vec))}
