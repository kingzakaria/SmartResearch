"""The labelled dataset for classification, built so that the classifier cannot read its own answer.

 * the label comes from the SUBJECT line (scripts/10_label_by_subject.py); the text given to the classifier is the body
   WITHOUT the subject lines that open it, and without the names of the panel members;
 * duplicates are left out, and decisions that share a template (a "family") get the same group id so that
   cross-validation never puts one in the training part and another in the test part;
 * categories that are too small are left out ("other" and anything under min_class_size).
"""
from collections import Counter
from dataclasses import dataclass, field

from ..extraction.cleaner import strip_panel_names
from ..morphology.normalizer import normalize_ar
from ..search.highlight import reading_text
from ..analysis.categories import UNLABELLED


@dataclass
class Dataset:
    ids: list[str]
    tokens: list[list[str]]                  # analysed text of each decision
    y: list[str]                             # category of each decision
    groups: list[str]                        # decisions of one group must stay on the same side of a split
    classes: list[str]
    left_out: dict = field(default_factory=dict)     # reason -> number of decisions left out


def classification_tokens(text: str, analyzer) -> list[str]:
    """Analysed stems without bare numbers (article numbers, dates, case numbers say nothing about the topic)."""
    return [t for t in analyzer.analyze(text) if not t.isdigit()]


def classification_text(doc: dict) -> str:
    """What the classifier is allowed to see: the reasoning, not the subject line and not the names."""
    return strip_panel_names(reading_text(doc.get("body", ""), doc.get("subject")))


def build_dataset(documents, labels: dict, drop=(), families=(), analyzer=None, min_class_size: int = 20) -> Dataset:
    drop = {str(d) for d in drop}
    family_of = {str(d): f"family{i}" for i, family in enumerate(families) for d in family}
    left_out: Counter = Counter()
    rows = []
    for doc in documents:
        doc_id = str(doc["doc_id"])
        if doc_id in drop:
            left_out["duplicate"] += 1
            continue
        category = labels.get(doc_id)
        if category in (None, UNLABELLED):
            left_out["no category"] += 1
            continue
        rows.append((doc_id, doc, category))
    sizes = Counter(c for _, _, c in rows)
    small = {c for c, n in sizes.items() if n < min_class_size}
    for _, _, c in rows:
        if c in small:
            left_out[f"class '{c}' too small"] += 1
    rows = [r for r in rows if r[2] not in small]
    return Dataset(
        ids=[r[0] for r in rows],
        tokens=[classification_tokens(classification_text(r[1]), analyzer) for r in rows],
        y=[r[2] for r in rows],
        groups=[family_of.get(r[0], f"doc{r[0]}") for r in rows],
        classes=sorted({r[2] for r in rows}),
        left_out=dict(left_out),
    )


def subject_leak_rate(documents, ids) -> tuple[float, list[str]]:
    """Share of the decisions whose subject (its first words) still appears in the text given to the classifier.
    0 means the subject lines were removed everywhere; anything above a few percent must be investigated."""
    by_id = {str(d["doc_id"]): d for d in documents}
    leaking = []
    for doc_id in ids:
        doc = by_id[doc_id]
        subject = normalize_ar(" ".join((doc.get("subject") or "").split()))
        if len(subject) < 12:
            continue
        probe = subject[:30].strip(" .-–—")
        if probe and probe in normalize_ar(" ".join(classification_text(doc).split())):
            leaking.append(doc_id)
    return (len(leaking) / len(ids) if ids else 0.0), leaking
