"""Pipeline A, step 2: structural cleaning of the raw text.

This is NOT linguistic normalization (that is Step 2, morphology/normalizer.py).
It only repairs encoding and layout problems so the metadata parser and the
RAG chunker work on a stable text. Words are never changed.
"""
import re

ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EXT_ARABIC_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Persian variants that appear in the corpus (e.g. "تحفیظ" with ی in 62.txt).
# They look identical on screen but are different characters, so words would never match.
PERSIAN_TO_ARABIC = str.maketrans({"ی": "ي", "ک": "ك", "ھ": "ه"})

# Invisible characters: BOM, zero-width space/joiners, bidi marks and embeddings.
INVISIBLE = re.compile(r"[﻿​-‏‪-‮⁦-⁩]")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = INVISIBLE.sub("", text)
    text = text.replace(" ", " ").replace("�", "")
    text = text.translate(ARABIC_INDIC).translate(EXT_ARABIC_INDIC).translate(PERSIAN_TO_ARABIC)
    text = re.sub(r"[ \t]+", " ", text)              # collapse spaces and tabs
    text = re.sub(r" ?\n ?", "\n", text)             # no spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)           # at most one blank line
    return text.strip()