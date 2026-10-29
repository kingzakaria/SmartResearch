"""Arabic-aware word tokenizer that retains Arabic and Western numerals."""

import re

from src.morphology.normalizer import normalize_arabic

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(normalize_arabic(text))
