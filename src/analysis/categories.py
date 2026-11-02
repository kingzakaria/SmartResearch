"""Give each decision a category from its SUBJECT line, with keyword rules (data/resources/category_rules.json).

Why the subject: it is a human-written summary of the topic, written separately from the reasoning. The classifier of
the next step is trained on the BODY only, so it cannot read its own label (no leakage).

Matching works on stems (same morphology as the rest of the project). The head of the subject (the text before
the first dash) is tried first, because it names the main issue; the whole subject is the second chance.
"""
import json
import re
from pathlib import Path

UNLABELLED = "other"
SEPARATOR = re.compile(r"\s[-–—]\s|[–—]")


def load_rules(path, analyzer) -> list[tuple[str, str, set[str]]]:
    """[(name, name_ar, {stems})] in priority order."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = []
    for category in data["categories"]:
        stems = {s for word in category["keywords"] for s in analyzer.analyze(word)}
        rules.append((category["name"], category.get("name_ar", category["name"]), stems))
    return rules


def subject_head(subject: str | None) -> str:
    """The part of the subject before the first dash: 'نزاع تحفيظ - عبء الإثبات' -> 'نزاع تحفيظ'."""
    if not subject:
        return ""
    parts = [p.strip(" .-–—") for p in SEPARATOR.split(subject)]
    return next((p for p in parts if p), "")


def assign_category(subject: str | None, rules, analyzer) -> tuple[str, str]:
    """(category, matched by) where matched by is 'head', 'subject' or 'none'."""
    if not subject:
        return UNLABELLED, "none"
    for scope, text in (("head", subject_head(subject)), ("subject", subject)):
        stems = set(analyzer.analyze(text))
        for name, _, keywords in rules:
            if stems & keywords:
                return name, scope
    return UNLABELLED, "none"


# ---------- reading the result of scripts/10_label_by_subject.py ----------
def read_labels(path) -> dict[str, str]:
    """doc_id -> category, from labels.csv ('other' decisions are not returned)."""
    import csv
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["doc_id"]: row["category"] for row in csv.DictReader(f) if row["category"] != UNLABELLED}


def propagate_to_duplicates(labels: dict[str, str], groups, drop) -> dict[str, str]:
    """A duplicate that was set aside is the same decision as its twin: it gets the twin's category."""
    labels = dict(labels)
    drop = {str(d) for d in drop}
    for group in groups:
        twin = next((labels[str(d)] for d in group if str(d) not in drop and str(d) in labels), None)
        for d in map(str, group):
            if d in drop and twin:
                labels[d] = twin
    return labels


def category_names(path) -> dict[str, str]:
    """category -> Arabic name, from category_rules.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {c["name"]: c.get("name_ar", c["name"]) for c in data["categories"]}
