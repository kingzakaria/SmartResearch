"""Pipeline C: the inverted index (the "map").

Instead of  document -> words  we store  word -> documents, with the occurrence count:

    "شفع"  -> {"12": 3, "40": 1, "97": 5}
    "عقار" -> {"12": 2, "97": 8, ...}

Looking a word up is then one dictionary access, not a scan of every document.
We also keep |d_j|, the number of terms of each document (needed by TF), and the document
frequency df(t) = number of documents containing t (needed by IDF).

doc_ids are strings everywhere (they are the keys of terms.json).
"""
import pickle
from collections import Counter
from pathlib import Path

INDEX_VERSION = 1


class InvertedIndex:
    def __init__(self):
        self.postings: dict[str, dict[str, int]] = {}    # term -> {doc_id: occurrences}
        self.doc_len: dict[str, int] = {}                 # doc_id -> |d_j| (number of terms)
        self.meta: dict = {}                              # how it was built, e.g. {"mode": "light"}

    # ---------- building ----------
    @classmethod
    def build(cls, terms_by_doc: dict) -> "InvertedIndex":
        index = cls()
        for doc_id, terms in terms_by_doc.items():
            index.add_document(doc_id, terms)
        return index

    def add_document(self, doc_id, terms: list[str]) -> None:
        doc_id = str(doc_id)
        if doc_id in self.doc_len:
            raise ValueError(f"document {doc_id!r} is already indexed")
        self.doc_len[doc_id] = len(terms)
        for term, count in Counter(terms).items():       # occurrence counting
            self.postings.setdefault(term, {})[doc_id] = count

    # ---------- reading ----------
    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @property
    def vocabulary(self) -> list[str]:
        return sorted(self.postings)

    def df(self, term: str) -> int:
        """Document frequency: in how many documents does the term occur?"""
        return len(self.postings.get(term, ()))

    def count(self, term: str, doc_id) -> int:
        """w(i,j): number of occurrences of the term in the document."""
        return self.postings.get(term, {}).get(str(doc_id), 0)

    def candidates(self, terms: list[str]) -> set[str]:
        """Documents containing AT LEAST ONE of the terms. Only these are worth scoring."""
        found: set[str] = set()
        for term in set(terms):
            found.update(self.postings.get(term, ()))
        return found

    @property
    def avg_doc_len(self) -> float:
        return sum(self.doc_len.values()) / self.n_docs if self.n_docs else 0.0

    # ---------- persistence ----------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"version": INDEX_VERSION, "postings": self.postings, "doc_len": self.doc_len,
                         "meta": self.meta}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "InvertedIndex":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if data.get("version") != INDEX_VERSION:
            raise ValueError(f"index file version {data.get('version')} is not supported "
                             f"(expected {INDEX_VERSION}): rebuild it with scripts/03_build_index.py")
        index = cls()
        index.postings, index.doc_len = data["postings"], data["doc_len"]
        index.meta = data.get("meta", {})                 # files written before Step 4 have no meta
        return index
