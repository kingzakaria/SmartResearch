"""Morphology step 3: stopwords loaded from EXTERNAL files (one word per line, '#' = comment).

Every entry goes through normalize_ar(), otherwise 'إلى' in the file would never match 'الي' in the text.
"""
from pathlib import Path

from .normalizer import normalize_ar

CLITIC_PREFIXES = ("و", "ف")      # وأن = و + أن ; فإن = ف + إن


def load_stopwords(*paths) -> set[str]:
    words = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Stopwords file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                words.add(normalize_ar(line))
    return words


def is_stopword(token: str, stopwords: set[str]) -> bool:
    """Exact match, or match after removing a one-letter conjunction prefix (و / ف)."""
    if token in stopwords:
        return True
    return len(token) > 2 and token[0] in CLITIC_PREFIXES and token[1:] in stopwords
