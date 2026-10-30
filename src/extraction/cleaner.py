"""Pipeline A, step 2: structural cleaning of the raw text.

This is NOT linguistic normalization (that is morphology/normalizer.py).
It only repairs encoding and layout problems so the metadata parser, the index and the
RAG chunker work on a stable text. Words are never changed.

Two layout defects of the PDF-to-text conversion are handled here:
  * letterhead lines (page headers) scattered anywhere in the text      -> removed
  * the closing ritual (panel, prosecutor, clerk, judges' names)        -> split_closing()
"""
import re

ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EXT_ARABIC_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Persian variants that appear in the corpus (e.g. "تحفیظ" with ی in 62.txt).
# They look identical on screen but are different characters, so words would never match.
PERSIAN_TO_ARABIC = str.maketrans({"ی": "ي", "ک": "ك", "ھ": "ه"})

# Invisible characters, built from code points so the source file contains no invisible text:
# BOM, zero-width space/joiners, bidi marks, bidi embeddings and isolates.
_INVISIBLE = [0xFEFF] + list(range(0x200B, 0x2010)) + list(range(0x202A, 0x202F)) + list(range(0x2066, 0x206A))
INVISIBLE = re.compile("[" + "".join(map(chr, _INVISIBLE)) + "]")
NBSP, REPLACEMENT_CHAR = chr(0x00A0), chr(0xFFFD)

# Page headers seen in the corpus. They sit on a line of their own, at ANY position
# (8.txt: before the operative part, 448.txt: after the signature block).
LETTERHEAD_LINES = {
    "المملكة المغربية",
    "المجلس الأعلى للسلطة القضائية",
    "محكمة النقض",
}

# "وبهذا صدر القرار وتلي بالجلسة العلنية ..." opens the closing ritual of every decision.
RE_CLOSING = re.compile(r"صدر\s+القرار\s+وتلي")
CLOSING_MAX_LINES = 15      # the real block is ~6 lines; a phrase far from the end is a quotation


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = INVISIBLE.sub("", text)
    text = text.replace(NBSP, " ").replace(REPLACEMENT_CHAR, "")
    text = text.translate(ARABIC_INDIC).translate(EXT_ARABIC_INDIC).translate(PERSIAN_TO_ARABIC)
    text = re.sub(r"[ \t]+", " ", text)              # collapse spaces and tabs
    text = re.sub(r" ?\n ?", "\n", text)             # no spaces around line breaks
    lines = [l for l in text.split("\n") if l.strip() not in LETTERHEAD_LINES]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)           # at most one blank line
    return text.strip()


def split_closing(text: str) -> tuple[str, str]:
    """Return (text without the closing ritual, the closing ritual itself).

    The block starts at the LAST line containing "صدر القرار وتلي" and runs to the end.
    If that line is more than CLOSING_MAX_LINES from the end, nothing is cut: the phrase is
    then part of the reasoning (for example a quotation of another decision)."""
    matches = list(RE_CLOSING.finditer(text))
    if not matches:
        return text, ""
    start = text.rfind("\n", 0, matches[-1].start()) + 1      # beginning of that line
    closing = text[start:]
    if closing.count("\n") + 1 > CLOSING_MAX_LINES:
        return text, ""
    return text[:start].strip(), closing.strip()
