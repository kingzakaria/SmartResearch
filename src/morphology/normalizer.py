"""Normalize common Arabic orthographic variants."""

import unicodedata


def normalize_arabic(text: str) -> str:
    normalized = []
    for char in text:
        if char == "\u0640" or unicodedata.category(char) == "Mn":
            continue
        normalized.append({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"}.get(char, char))
    return unicodedata.normalize("NFC", "".join(normalized))
