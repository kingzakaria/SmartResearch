"""Morphology step 1: orthographic normalization of Arabic text.

Goal: different spellings of the same word become ONE string, so they match in the index.
Order matters: the stopword files go through this same function.
"""
import re

DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟ]")   # tashkeel, small marks
TATWEEL = "ـ"
ALEF_VARIANTS = re.compile("[إأآٱ]")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
PERSIAN = str.maketrans({"ی": "ي", "ک": "ك", "ھ": "ه"})


def normalize_ar(text: str, ta_marbuta: bool = True) -> str:
    """
    - remove diacritics and tatweel          كَتَبَ -> كتب ,  كـــتاب -> كتاب
    - unify alef forms                        أ إ آ ٱ -> ا
    - alef maqsura -> ya                      على -> علي
    - ta marbuta -> ha (optional)             محكمة -> محكمه   (recall over precision)
    - Arabic-Indic digits -> 0-9
    """
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = ALEF_VARIANTS.sub("ا", text)
    text = text.replace("ى", "ي")
    if ta_marbuta:
        text = text.replace("ة", "ه")
    return text.translate(DIGITS).translate(PERSIAN)
