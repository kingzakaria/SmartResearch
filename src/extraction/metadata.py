"""Pipeline A, step 3: parse the header block of a Court of Cassation decision.

Typical header (8.txt):
    قرار محكمة النقض
    رقم 1/15
    الصادر بتاريخ 21 مارس 2023
    في الملف العقاري رقم 2020/1/1/4066
    <subject line, sometimes wrapped over several lines, usually ending with ".">
    <reasoning ...>

Variants observed in the corpus and handled here:
  * decision number: "رقم 1/15" or "رقم 168" (no slash)
  * date: "28مارس 2023" (no space), "ابريل" without hamza, typo "الصاور"
  * case number: separators / | \\ or none ("2020115399"), typo "علف عقاري"
  * some files have NO header at all (62.txt starts with "وبعد المداولة")
"""
import re

from .cleaner import clean_text

HEADER_LINES = 6          # metadata always sits in the first lines of the file
SUBJECT_MAX_LINES = 6     # a wrapped subject line is never longer than this

# Month names, keyed WITHOUT hamza/diacritics so spelling variants all match.
MONTHS = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "ماي": 5, "مايو": 5,
    "يونيو": 6, "يوليوز": 7, "يوليو": 7, "غشت": 8, "اغسطس": 8,
    "شتنبر": 9, "سبتمبر": 9, "اكتوبر": 10, "نونبر": 11, "نوفمبر": 11,
    "دجنبر": 12, "ديسمبر": 12,
}

RE_DECISION = re.compile(r"^\s*(?:رقم|عدد)\s*(\d{1,5}(?:\s*/\s*\d{1,5})?)\s*$")
RE_DATE = re.compile(r"بتاريخ\s*(\d{1,2})\s*([ء-ي]+)\s*(\d{4})")
RE_CASE = re.compile(r"(?:الملف|ملف|علف)\s*(?:العقاري|عقاري)?\s*(?:رقم|عدد)\s*(\d[\d/|\\\-]*)")
RE_CASE_FALLBACK = re.compile(r"(?:رقم|عدد)\s*(\d[\d/|\\\-]{5,})")   # for OCR-like typos of "الملف"


def _strip_alef(word: str) -> str:
    return re.sub("[إأآٱ]", "ا", word)


def _normalize_case_no(raw: str) -> str:
    return re.sub(r"[|\\\-]", "/", raw).strip("/")


def _parse_date(line: str):
    m = RE_DATE.search(line)
    if not m:
        return None
    day, month_word, year = int(m.group(1)), _strip_alef(m.group(2)), int(m.group(3))
    month = MONTHS.get(month_word)
    if not month or not 1 <= day <= 31:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_header(text: str) -> dict:
    """Return metadata and the body of one (already cleaned) decision."""
    lines = text.split("\n")
    head = lines[:HEADER_LINES]

    decision_no = date = case_no = None
    decision_idx = date_idx = case_idx = None

    for i, line in enumerate(head):
        if decision_no is None and (m := RE_DECISION.match(line)):
            decision_no, decision_idx = re.sub(r"\s+", "", m.group(1)), i
        elif date is None and "بتاريخ" in line and (d := _parse_date(line)):
            date, date_idx = d, i
        elif case_no is None and (m := RE_CASE.search(line)):
            case_no, case_idx = _normalize_case_no(m.group(1)), i

    if case_no is None:   # fallback: a header line (not the date line) holding a long number
        for i, line in enumerate(head):
            if i in (decision_idx, date_idx) or "بتاريخ" in line:
                continue
            if i > 0 and (m := RE_CASE_FALLBACK.search(line)):
                case_no, case_idx = _normalize_case_no(m.group(1)), i
                break

    found = [i for i in (decision_idx, date_idx, case_idx) if i is not None]
    subject, body_start = None, 0
    if found:
        start = max(found) + 1
        chunk = []
        for line in lines[start:start + SUBJECT_MAX_LINES]:
            chunk.append(line.strip())
            if line.rstrip().endswith("."):          # the subject ends with a full stop
                break
        else:
            chunk = chunk[:1]                         # no full stop found: keep the first line only
        subject = re.sub(r"\s*\n\s*", " ", " ".join(c for c in chunk if c)).strip() or None
        body_start = start

    body = "\n".join(lines[body_start:]).strip()
    return {
        "decision_no": decision_no,
        "date": date,
        "year": int(date[:4]) if date else None,
        "case_no": case_no,
        "subject": subject,
        "body": body,
    }


def parse_document(raw_text: str) -> dict:
    """Clean + parse. Returns the metadata plus the cleaned full text."""
    text = clean_text(raw_text)
    meta = parse_header(text)
    meta["text"] = text
    return meta