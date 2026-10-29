"""Load and normalize stopwords from UTF-8 resource files."""

from pathlib import Path

from src.morphology.normalizer import normalize_arabic
from src.morphology.tokenizer import tokenize


def load_stopwords(*paths: Path | str) -> set[str]:
    words: set[str] = set()
    for source in paths:
        path = Path(source)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            words.update(tokenize(normalize_arabic(line.split("#", 1)[0])))
    return words
