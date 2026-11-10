"""Naive Bayes for text, written by hand with the formulas of the course (Apprentissage supervise) and of the TP.

    P(c)     = number of documents of class c / number of documents
    P(w|c)   = ( n(w,c) + 1 ) / ( N_c + m )          Laplace smoothing
    c_MAP    = argmax over c of  P(c) x PRODUCT over the words w of d of P(w|c)

    n(w,c)  occurrences of w in the documents of class c      N_c  words (with repetitions) in class c
    m       distinct words of the whole training corpus

Two practical points, both from the TP:
  * the product of hundreds of small probabilities underflows to 0.0, so the SUM OF LOGARITHMS is used:
        log P(c) + sum log P(w|c)        (same winner, since log(a.b) = log a + log b)
  * a word that is not in the vocabulary is ignored.

The model stores the COUNTS n(w,c), not the probabilities: P(w|c) is computed on demand from the formula, which
keeps the file small (a class does not list the whole vocabulary) and gives exactly the same numbers.
"""
import json
import math
from collections import Counter
from pathlib import Path


class NaiveBayes:
    def __init__(self):
        self.classes: list[str] = []
        self.n_docs: dict[str, int] = {}              # documents per class
        self.counts: dict[str, Counter] = {}          # n(w, c)
        self.totals: dict[str, int] = {}              # N_c
        self.vocabulary: set[str] = set()             # its size is m

    # ---------- training ----------
    def fit(self, documents: list[list[str]], labels: list[str]) -> "NaiveBayes":
        if len(documents) != len(labels):
            raise ValueError("documents and labels must have the same length")
        if len(set(labels)) < 2:
            raise ValueError("Naive Bayes needs at least two classes")
        self.classes = sorted(set(labels))
        self.n_docs = {c: 0 for c in self.classes}
        self.counts = {c: Counter() for c in self.classes}
        for tokens, label in zip(documents, labels):
            self.n_docs[label] += 1
            self.counts[label].update(tokens)
        self.totals = {c: sum(self.counts[c].values()) for c in self.classes}
        self.vocabulary = set().union(*(set(c) for c in self.counts.values()))
        return self

    # ---------- the formulas ----------
    @property
    def m(self) -> int:
        return len(self.vocabulary)

    def prior(self, c: str) -> float:
        return self.n_docs[c] / sum(self.n_docs.values())

    def p_word(self, word: str, c: str) -> float:
        """P(w|c) with Laplace smoothing. Over the whole vocabulary these probabilities sum to 1."""
        return (self.counts[c].get(word, 0) + 1) / (self.totals[c] + self.m)

    def log_scores(self, tokens: list[str], uniform_priors: bool = False) -> dict[str, float]:
        """log P(c) + sum of log P(w|c). With uniform_priors every class starts equal (P(c) = 1/number of classes):
        useful when the classes are very unbalanced and a query should not lose just because its chamber is small."""
        known = [t for t in tokens if t in self.vocabulary]
        return {c: math.log(1 / len(self.classes) if uniform_priors else self.prior(c))
                + sum(math.log(self.p_word(t, c)) for t in known) for c in self.classes}

    def n_known(self, tokens: list[str]) -> int:
        return sum(1 for t in tokens if t in self.vocabulary)

    def predict_proba(self, tokens: list[str], uniform_priors: bool = False) -> dict[str, float]:
        """The scores turned into percentages that sum to 1 (softmax of the log scores, stable for large negatives).
        Read them as "which class wins", not as a real certainty: the independence assumption makes them over-confident."""
        scores = self.log_scores(tokens, uniform_priors)
        top = max(scores.values())
        exp = {c: math.exp(s - top) for c, s in scores.items()}
        total = sum(exp.values())
        return {c: v / total for c, v in exp.items()}

    def predict(self, tokens: list[str], uniform_priors: bool = False) -> str:
        scores = self.log_scores(tokens, uniform_priors)
        return max(self.classes, key=lambda c: (scores[c], c))

    # ---------- persistence ----------
    def to_dict(self) -> dict:
        return {"classes": self.classes, "n_docs": self.n_docs, "totals": self.totals,
                "counts": {c: dict(self.counts[c]) for c in self.classes}}

    @classmethod
    def from_dict(cls, data: dict) -> "NaiveBayes":
        model = cls()
        model.classes = list(data["classes"])
        model.n_docs = {c: int(n) for c, n in data["n_docs"].items()}
        model.totals = {c: int(n) for c, n in data["totals"].items()}
        model.counts = {c: Counter(data["counts"][c]) for c in model.classes}
        model.vocabulary = set().union(*(set(c) for c in model.counts.values()))
        return model

    def save(self, path, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"meta": meta or {}, **self.to_dict()}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> tuple["NaiveBayes", dict]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data), data.get("meta", {})
