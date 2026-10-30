"""Morphology API: ONE entry point for all text processing (documents AND queries).

    analyzer = ArabicMorphologyAnalyzer(generic_files, legal_files, mode="light")
    analyzer.analyze("المحكمة الطاعنين الشفعة")  ->  ["شفع"]  (boilerplate removed, rest stemmed)

Pipeline:  normalize -> tokenize -> stopwords -> stem -> drop boilerplate stems

Rule 1 of the project: documents at indexing time and queries at search time MUST go through the
same analyze(), otherwise their terms never match.
"""
from .normalizer import normalize_ar
from .stemmer import Stemmer
from .stopwords import is_stopword, load_stopwords
from .tokenizer import tokenize


class ArabicMorphologyAnalyzer:
    def __init__(self, generic_files=(), legal_files=(), mode: str = "light",
                 ta_marbuta: bool = True):
        self.mode = mode
        self.ta_marbuta = ta_marbuta
        self.stemmer = Stemmer(mode)
        self.generic = load_stopwords(*generic_files) if generic_files else set()
        self.legal = load_stopwords(*legal_files) if legal_files else set()
        self.stopwords = self.generic | self.legal
        # boilerplate is also removed in its stemmed form (الطاعنين -> طاعن), see stopwords_legal.txt
        self.legal_stems = {self.stemmer.stem(w) for w in self.legal}

    # ---- the three steps, exposed separately for tests and for the report ----
    def normalize(self, text: str) -> str:
        return normalize_ar(text, self.ta_marbuta)

    def tokens(self, text: str) -> list[str]:
        """Normalized tokens, before stopword removal and stemming."""
        return tokenize(self.normalize(text))

    def analyze(self, text: str) -> list[str]:
        terms = []
        for tok in self.tokens(text):
            if is_stopword(tok, self.stopwords):
                continue
            stem = self.stemmer.stem(tok)
            if len(stem) < 2 or stem in self.legal_stems:
                continue
            terms.append(stem)
        return terms

    def explain(self, text: str) -> list[tuple]:
        """(token, status, stem) for every token: shows exactly what the pipeline did."""
        rows = []
        for tok in self.tokens(text):
            if is_stopword(tok, self.stopwords):
                rows.append((tok, "stopword", ""))
                continue
            stem = self.stemmer.stem(tok)
            if len(stem) < 2 or stem in self.legal_stems:
                rows.append((tok, "boilerplate", stem))
            else:
                rows.append((tok, "kept", stem))
        return rows


def analyzer_from_config(config, mode: str | None = None) -> ArabicMorphologyAnalyzer:
    return ArabicMorphologyAnalyzer(
        generic_files=[config.STOPWORDS_GENERIC],
        legal_files=[config.STOPWORDS_LEGAL],
        mode=mode or config.ANALYZER_MODE,
    )
