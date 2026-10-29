"""Composable Arabic text analyzer shared by indexing and query search."""

from pathlib import Path
from collections.abc import Iterable

from config import DEFAULT_STEMMING
from src.morphology.normalizer import normalize_arabic
from src.morphology.stopwords import load_stopwords
from src.morphology.stemmer import ArabicStemmer
from src.morphology.tokenizer import tokenize


class ArabicMorphologyAnalyzer:
    def __init__(
        self,
        stopword_paths: Iterable[Path | str] = (),
        stemming: str | None = DEFAULT_STEMMING,
        extra_stopwords: Iterable[str] = (),
    ) -> None:
        self.stopwords = load_stopwords(*stopword_paths)
        self.stopwords.update(normalize_arabic(word) for word in extra_stopwords)
        self.stemmer = ArabicStemmer(stemming) if stemming else None

    def analyze(self, text: str) -> list[str]:
        terms = []
        for word in tokenize(text):
            if word in self.stopwords:
                continue
            term = self.stemmer.stem(word) if self.stemmer else word
            if term:
                terms.append(term)
        return terms
