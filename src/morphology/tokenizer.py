"""Morphology step 2: tokenization.

Splits NORMALIZED text into words and removes punctuation (. ، ؛ ؟ ! : ( ) " ...).
Numbers are kept only when short: article numbers such as "371" or "43" carry legal meaning,
but dates, years and file numbers (2005, 4066, 2020115399) are noise.
"""
import re

TOKEN = re.compile(r"[ء-ي]+|\d+")     # Arabic letters, or digits. Everything else is dropped.

MIN_WORD_LEN = 2
MIN_DIGITS, MAX_DIGITS = 2, 3


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in TOKEN.findall(text):
        if tok.isdigit():
            if MIN_DIGITS <= len(tok) <= MAX_DIGITS:
                tokens.append(tok)
        elif len(tok) >= MIN_WORD_LEN:
            tokens.append(tok)
    return tokens
