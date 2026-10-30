"""Highlight literal query terms in plain-text results."""

import re


def highlight_matches(text: str, terms: list[str], start: str = "**", end: str = "**") -> str:
    unique = sorted({term for term in terms if term}, key=len, reverse=True)
    if not unique:
        return text
    pattern = re.compile("(" + "|".join(re.escape(term) for term in unique) + ")", re.IGNORECASE)
    return pattern.sub(lambda match: f"{start}{match.group(0)}{end}", text)
