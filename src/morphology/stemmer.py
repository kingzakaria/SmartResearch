"""Conservative light stemming with optional NLTK ISRI support."""

from src.morphology.normalizer import normalize_arabic

_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ك", "ل")
_SUFFIXES = ("يات", "يون", "يين", "ات", "ان", "ون", "ين", "ها", "هم", "هن", "كم", "كن", "ية", "ه", "ة", "ي", "ك")


def light_stem(word: str) -> str:
    stem = normalize_arabic(word)
    for prefix in _PREFIXES:
        if stem.startswith(prefix) and len(stem) - len(prefix) >= 3:
            stem = stem[len(prefix):]
            break
    for suffix in _SUFFIXES:
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 3:
            stem = stem[:-len(suffix)]
            break
    return stem


class ArabicStemmer:
    def __init__(self, mode: str = "light") -> None:
        if mode not in {"light", "isri"}:
            raise ValueError("mode must be 'light' or 'isri'")
        self.mode = mode
        self._isri = None
        if mode == "isri":
            try:
                from nltk.stem.isri import ISRIStemmer
            except ImportError as exc:
                raise RuntimeError("ISRI mode requires NLTK; install it with: pip install nltk") from exc
            self._isri = ISRIStemmer()

    def stem(self, word: str) -> str:
        normalized = normalize_arabic(word)
        return light_stem(normalized) if self._isri is None else self._isri.stem(normalized)
