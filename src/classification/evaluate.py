"""Cross-validation and held-out classifier evaluation."""

from collections.abc import Mapping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import DEFAULT_CV_FOLDS


def evaluate_classifier(
    model: Pipeline,
    documents: Mapping[str, list[str]],
    labels: Mapping[str, str],
    folds: int = DEFAULT_CV_FOLDS,
) -> dict[str, object]:
    ids = [doc_id for doc_id in documents if doc_id in labels]
    y = [labels[doc_id] for doc_id in ids]
    texts = [" ".join(documents[doc_id]) for doc_id in ids]
    class_counts = {label: y.count(label) for label in set(y)}
    split_count = min(folds, min(class_counts.values(), default=0))
    if split_count < 2:
        raise ValueError("cross-validation requires at least two examples per class")
    predictions = cross_val_predict(
        model, texts, y, cv=StratifiedKFold(n_splits=split_count, shuffle=True, random_state=42)
    )
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(y, predictions, zero_division=0, output_dict=True),
    }
