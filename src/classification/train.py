"""Train a text classifier from analyzed documents and CSV labels."""

from collections.abc import Mapping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import math

from config import DEFAULT_CV_FOLDS

_CLASSIFIERS = {
    "naive_bayes": MultinomialNB,
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "svm": LinearSVC,
    "knn": lambda: KNeighborsClassifier(n_neighbors=3),
}


def train_classifier(
    documents: Mapping[str, list[str]], labels: Mapping[str, str], algorithm: str = "naive_bayes"
) -> Pipeline:
    if algorithm not in _CLASSIFIERS:
        raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {', '.join(_CLASSIFIERS)}")
    pairs = [(doc_id, labels[doc_id]) for doc_id in documents if doc_id in labels]
    if len({label for _, label in pairs}) < 2:
        raise ValueError("classifier training requires at least two labeled classes")
    if len(pairs) < 2:
        raise ValueError("classifier training requires at least two labeled documents")
    classifier = _CLASSIFIERS[algorithm]()
    if algorithm == "knn":
        classes = {category for _, category in pairs}
        class_counts = [sum(category == candidate for _, category in pairs) for candidate in classes]
        folds = min(DEFAULT_CV_FOLDS, min(class_counts))
        training_size = len(pairs) - math.ceil(len(pairs) / folds) if folds > 1 else len(pairs)
        neighbors = min(3, training_size)
        if neighbors % 2 == 0:
            neighbors -= 1
        classifier.set_params(n_neighbors=max(1, neighbors))
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")),
            ("classifier", classifier),
        ]
    )
    model.fit([" ".join(documents[doc_id]) for doc_id, _ in pairs], [label for _, label in pairs])
    return model
