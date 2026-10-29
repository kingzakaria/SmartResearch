"""A lightweight TF-IDF text embedder suitable for offline retrieval."""

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        if not texts or not any(text.strip() for text in texts):
            raise ValueError("at least one non-empty text is required to fit embeddings")
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list[str]):
        if not self._fitted:
            raise RuntimeError("embedder must be fitted before encoding")
        return self.vectorizer.transform(texts)
