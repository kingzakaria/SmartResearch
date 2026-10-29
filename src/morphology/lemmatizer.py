"""Optional Qalsadi-backed Arabic lemmatizer."""


class ArabicLemmatizer:
    def __init__(self) -> None:
        try:
            import qalsadi.lemmatizer
        except ImportError as exc:
            raise RuntimeError("Qalsadi is optional; install it with: pip install qalsadi") from exc
        self._lemmatizer = qalsadi.lemmatizer.Lemmatizer()

    def lemmatize(self, word: str) -> str:
        return self._lemmatizer.lemmatize(word)
