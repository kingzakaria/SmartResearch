"""Find the query words inside a document and build a snippet with the matches marked.

The index works on STEMS (شفع), but the user must see the ORIGINAL words of the text (بالشفعة).
So every word of the document is run through the same analyzer as the query, and it is marked
when its stem is one of the query stems.
"""
import re

# Arabic letters and combining marks (diacritics, tatweel), or digits. Built from code points so the
# source holds no invisible characters.
WORD = re.compile("[" + chr(0x0621) + "-" + chr(0x065F) + chr(0x0670) + r"]+|\d+")
LEFT, RIGHT = "«", "»"


def find_matches(text: str, query_terms, analyzer, cache: dict | None = None) -> list[tuple[int, int]]:
    """(start, end) of every word of `text` whose stem is one of the query terms."""
    wanted = set(query_terms)
    cache = {} if cache is None else cache
    spans = []
    for m in WORD.finditer(text):
        word = m.group()
        if word not in cache:
            stems = analyzer.analyze(word)
            cache[word] = stems[0] if stems else None
        if cache[word] in wanted:
            spans.append(m.span())
    return spans


def make_snippet(text: str, spans: list[tuple[int, int]], width: int = 280,
                 left: str = LEFT, right: str = RIGHT) -> str:
    """The window of `width` characters that contains the most matches, matches wrapped in markers."""
    if not text:
        return ""
    if not spans:
        return re.sub(r"\s+", " ", text[:width]).strip() + (" …" if len(text) > width else "")

    best_start, best_count = spans[0][0], 0
    for s, _ in spans:                                   # window starting at each match
        count = sum(1 for s2, _ in spans if s <= s2 < s + width)
        if count > best_count:
            best_start, best_count = s, count

    start = max(0, best_start - 40)
    while start > 0 and not text[start - 1].isspace():   # do not start in the middle of a word
        start -= 1
    end = min(len(text), start + width)
    while end < len(text) and not text[end].isspace():   # do not stop in the middle of a word
        end += 1

    pieces, cursor = [], start
    for s, e in spans:
        if s >= start and e <= end:
            pieces += [text[cursor:s], left, text[s:e], right]
            cursor = e
    pieces.append(text[cursor:end])
    snippet = re.sub(r"\s+", " ", "".join(pieces)).strip()
    return ("… " if start > 0 else "") + snippet + (" …" if end < len(text) else "")


def segment_text(text: str, spans: list[tuple[int, int]]) -> list[dict]:
    """Cut `text` into [{"t": text, "m": is_match}, ...] so a client can mark the matches
    without computing offsets itself (JavaScript and Python count characters differently)."""
    segments, cursor = [], 0
    for start, end in spans:
        if start < cursor:
            continue
        if start > cursor:
            segments.append({"t": text[cursor:start], "m": False})
        segments.append({"t": text[start:end], "m": True})
        cursor = end
    if cursor < len(text):
        segments.append({"t": text[cursor:], "m": False})
    return segments


def reading_text(body: str, subject: str | None = None) -> str:
    """The body prepared for reading: the subject lines that open the body are dropped (the page shows
    the subject as its title) and the line breaks of the PDF are turned into spaces, keeping blank
    lines as paragraph breaks."""
    lines = body.split("\n")
    if subject:
        wanted, collected, taken = re.sub(r"\s+", " ", subject).strip(), [], 0
        for i, line in enumerate(lines):
            if not collected and re.fullmatch(r"[\s\-–—]*", line):
                taken = i + 1                                   # leading "-" lines
                continue
            collected.append(line.strip())
            if re.sub(r"\s+", " ", " ".join(collected)).strip() == wanted:
                lines = lines[i + 1:]
                break
            if len(" ".join(collected)) > len(wanted) + 5:
                break
    text = "\n".join(lines).strip()
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)
