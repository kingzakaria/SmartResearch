"""Pipeline A, step 3: parse the header block of a Court of Cassation decision.

Typical header (8.txt):
    قرار محكمة النقض
    رقم 1/15
    الصادر بتاريخ 21 مارس 2023
    في الملف العقاري رقم 2020/1/1/4066
    <subject line, sometimes wrapped over several lines>
    <reasoning ...>

Variants observed in the corpus and handled here:
  * decision number: "رقم 1/15", "رقم 168" (no slash), typo "رتم 3/41"
  * date: "28مارس 2023" (no space), "ابريل" without hamza, typo "الصاور"
  * case number: separators / | \\ or none ("2020115399"), typo "علف عقاري",
    no word "رقم" ("في الملف العقاري 2019/4/1/7471"),
    extra words ("ملف عقاري - الهيئة الثالثة - رقم 2021/8/1/750")
  * subject BEFORE the title line "قرار محكمة النقض" (642.txt, 639.txt)
  * subject without a final full stop, followed directly by the reasoning
  * files with NO header at all (62.txt starts with "وبعد المداولة")
"""
import re

from .cleaner import clean_text, split_closing

TITLE = "قرار محكمة النقض"
TITLE_SEARCH_LINES = 10   # the title line is within the first lines of the file
HEADER_SPAN = 7           # metadata lines that follow the title line
SUBJECT_MAX_LINES = 6     # a wrapped subject is never longer than this
SHORT_LINE = 30           # a short fragment ("بيع", "رسم") belongs to a wrapped subject

# Month names, keyed WITHOUT hamza/diacritics so spelling variants all match.
MONTHS = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "ماي": 5, "مايو": 5,
    "يونيو": 6, "يوليوز": 7, "يوليو": 7, "غشت": 8, "اغسطس": 8,
    "شتنبر": 9, "سبتمبر": 9, "اكتوبر": 10, "نونبر": 11, "نوفمبر": 11,
    "دجنبر": 12, "ديسمبر": 12,
}

# the label is any 3-letter word (رقم / رتم / عدد) alone on its line with the number
RE_DECISION = re.compile(r"^\s*[ء-ي]{3}\s*(\d{1,5}(?:\s*/\s*\d{1,5})?)\s*$")
RE_DATE = re.compile(r"بتاريخ\s*(\d{1,2})\s*([ء-ي]+)\s*(\d{4})")
# label (الملف / ملف / علف), anything that is not a digit, then the number itself
RE_CASE = re.compile(r"(?:الملف|ملف|علف)[^\d\n]*?(\d[\d/|\\\-]{5,})")
# fallbacks, used only when a real header exists
RE_CASE_FALLBACK = re.compile(r"(?:رقم|عدد)\s*(\d[\d/|\\\-]{5,})")
RE_CASE_SHAPE = re.compile(r"(?<![\d/])(\d{1,4}[/|\\]\d{1,2}[/|\\]\d{1,2}[/|\\]\d{1,5})(?![\d/])")

BODY_STARTERS = ("إن ", "ان ", "لما ", "حيث ", "المقرر", "الثابت", "لكن ", "وفقا ", "بمقتضى")
DASH_ONLY = re.compile(r"^[\s\-–—]*$")
SPACED_DASH = re.compile(r"\s[-–—]\s")


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


def _is_subject_line(line: str) -> bool:
    """A subject line is a short fragment or a heading with a spaced dash ('A - B – أثره').
    Reasoning lines are long prose without such dashes, so they stop the subject."""
    s = line.strip()
    if not s or s.startswith(BODY_STARTERS):
        return False
    return len(s) <= SHORT_LINE or bool(SPACED_DASH.search(s))


def _collect_subject(lines: list) -> str | None:
    """Read subject lines from the start of `lines` until the reasoning begins."""
    chunk = []
    for line in lines[:SUBJECT_MAX_LINES]:
        if not chunk and DASH_ONLY.match(line):
            continue                                   # leading "-" separator lines
        if not _is_subject_line(line):
            break
        chunk.append(line.strip())
        if line.rstrip().endswith("."):                # a full stop ends the subject
            break
    while chunk and DASH_ONLY.match(chunk[-1]):
        chunk.pop()                                    # trailing "-" lines
    return " ".join(chunk).strip() or None


def parse_header(text: str) -> dict:
    """Return metadata and the body of one (already cleaned) decision."""
    lines = text.split("\n")

    anchor = next((i for i, l in enumerate(lines[:TITLE_SEARCH_LINES])
                   if l.strip().startswith(TITLE) and len(l.strip()) < 30), None)
    start_scan = anchor if anchor is not None else 0
    window = range(start_scan, min(len(lines), start_scan + HEADER_SPAN))

    decision_no = date = case_no = None
    used = {}                                          # field -> line index

    for i in window:
        line = lines[i]
        if decision_no is None and (m := RE_DECISION.match(line)):
            decision_no, used["decision"] = re.sub(r"\s+", "", m.group(1)), i
        elif date is None and "بتاريخ" in line and (d := _parse_date(line)):
            date, used["date"] = d, i
        elif case_no is None and (m := RE_CASE.search(line)):
            case_no, used["case"] = _normalize_case_no(m.group(1)), i

    header_exists = anchor is not None or bool(used)
    if case_no is None and header_exists:              # fallbacks, never on a header-less file
        for i in window:
            line = lines[i]
            if i in used.values() or "بتاريخ" in line or i == anchor:
                continue
            m = RE_CASE_FALLBACK.search(line) or RE_CASE_SHAPE.search(line)
            if m:
                case_no, used["case"] = _normalize_case_no(m.group(1)), i
                break

    subject, body_start, pre_subject = None, 0, None
    if used:
        # a subject written BEFORE the title line (642.txt, 639.txt)
        if anchor:
            pre_subject = _collect_subject(lines[:anchor])
        body_start = max(used.values()) + 1
        post_subject = _collect_subject(lines[body_start:])
        subject = " ".join(s for s in (pre_subject, post_subject) if s) or None

    body = "\n".join(lines[body_start:]).strip()
    if pre_subject:
        body = pre_subject + "\n" + body
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
    meta["body"], meta["closing"] = split_closing(meta["body"])   # index text excludes the closing ritual
    meta["text"] = text                                           # full cleaned text, kept for display
    return meta
