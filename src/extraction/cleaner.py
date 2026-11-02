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


# ---------- names of the court members ----------
# "وبعد تلاوة المستشار المقرر السيد <name> لتقريره والاستماع إلى ملاحظات المحامي العام السيد <name> الرامية إلى رفض الطلب"
# The names identify the panel and the session, not the subject: left in, they would be "the most characteristic words" of a
# group of decisions. They are removed from the text used for ANALYSIS only (the text shown to the reader is untouched);
# the useful words around them ("الرامية إلى رفض الطلب") stay.
_NAME = r"(?:[ء-ي]+[\s،,]+){0,5}?[ء-ي]+"
_AFTER_RAPPORTEUR = r"(?=\s+(?:لتقريره|وتقديم|وبعد|والاستماع|ولملاحظات|وبمحضر|وبمساعدة|الذي)\b)"
_PANEL_PATTERNS = [
    # rapporteur: "المستشار المقرر السيد X لتقريره"  /  "... السيد X وتقديم المحامي العام"
    (re.compile(r"(المستشار(?:ة)?\s+المقرر(?:ة)?\s+)(?:السيد(?:ة)?\s+)" + _NAME + _AFTER_RAPPORTEUR), r"\1"),
    # advocate general: "المحامي العام السيد X الرامية ... / وبمساعدة ..."
    (re.compile(r"(المحامي(?:ة)?\s+العام(?:ة)?\s+)(?:السيد(?:ة)?\s+)" + _NAME +
                r"(?=\s+(?:الرامية|الرامي|الهادفة|الذي|التي|وبمساعدة|وبحضور|وبعد|ولملاحظات|ملاحظاته|ملتمسه|ملتمساته|مستنتجاته)\b)"), r"\1"),
    # clerk: "وبمساعدة كاتبة الضبط السيدة X."
    (re.compile(r"(وبمساعدة\s+كاتب(?:ة)?\s+الضبط\s+)(?:السيد(?:ة)?\s+)?" + _NAME + r"(?=\s*(?:[.،]|$))"), r"\1"),
    # composition: "<name> رئيسا والمستشارين السادة[:] X ،مقررا، Y ... أعضاء"
    (re.compile(r"(?:[ء-ي]+[\s،,]+){0,4}?رئيسا\s+(?:و\s*)?المستشار(?:ين|ات)\s+(?:السادة|السيدات)\s*:?\s*"
                r"(?:[ء-ي]+[\s،,]+){1,25}?[أا]عضاء"), ""),
]


def strip_panel_names(text: str) -> str:
    """The text without the names of the panel members, prosecutor and clerk (for indexing and clustering)."""
    for pattern, replacement in _PANEL_PATTERNS:
        text = pattern.sub(replacement, text)
    return re.sub(r"[ \t]{2,}", " ", text)
