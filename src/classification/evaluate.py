"""Classifiers and cross-validation (scikit-learn).

Pipeline, fitted on the training part of each fold ONLY (so nothing leaks from the test part):
    tf-idf with frequency thresholding (min_df / max_df)  ->  chi-square keeps the best X% of the terms  ->  classifier

Cross-validation: 5 folds, stratified (every fold has every category) and grouped (a duplicate or a template family is
never split between training and test). It is repeated with different random splits and averaged.
"""
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def identity(tokens):
    """The documents are already analysed (list of stems): the vectorizer must not tokenise them again."""
    return tokens


MODELS = {
    "majority": lambda: DummyClassifier(strategy="most_frequent"),
    "naive_bayes": lambda: ComplementNB(alpha=0.3),
    "rocchio": lambda: NearestCentroid(),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance", algorithm="brute"),
    "logistic_regression": lambda: LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
    "linear_svm": lambda: LinearSVC(C=0.5, class_weight="balanced"),
}


def make_pipeline(model: str, percentile: int = 25, min_df: int = 2, max_df: float = 0.5) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer=identity, min_df=min_df, max_df=max_df, sublinear_tf=True)),
        ("chi2", SelectPercentile(chi2, percentile=percentile)),
        ("model", MODELS[model]()),
    ])


def cross_validate(dataset, model: str, percentile: int = 25, min_df: int = 2, max_df: float = 0.5,
                   n_splits: int = 5, repeats: int = 3, seed: int = 42) -> dict:
    """Mean and standard deviation of accuracy and macro-F1 over folds and repeats, plus the pooled predictions
    of the first repeat (each decision predicted by a model that never saw it)."""
    y = np.array(dataset.y)
    accuracies, f1s, pooled = [], [], None
    for r in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        predicted = np.empty(len(y), dtype=object)
        for train, test in splitter.split(np.zeros(len(y)), y, dataset.groups):
            pipe = make_pipeline(model, percentile, min_df, max_df)
            pipe.fit([dataset.tokens[i] for i in train], y[train])
            predicted[test] = pipe.predict([dataset.tokens[i] for i in test])
            accuracies.append(accuracy_score(y[test], predicted[test]))
            f1s.append(f1_score(y[test], predicted[test], average="macro", zero_division=0))
        if pooled is None:
            pooled = predicted.copy()
    return {"model": model, "percentile": percentile, "min_df": min_df, "max_df": max_df,
            "accuracy": float(np.mean(accuracies)), "accuracy_std": float(np.std(accuracies)),
            "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)), "pooled": list(pooled)}


def folds_keep_groups_together(dataset, n_splits: int = 5, seed: int = 42) -> bool:
    """Check used by the tests: no group has members in two different test folds."""
    y = np.array(dataset.y)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of_group: dict = {}
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(y)), y, dataset.groups)):
        for i in test:
            if fold_of_group.setdefault(dataset.groups[i], k) != k:
                return False
    return True


def top_terms_per_class(pipeline: Pipeline, classes, n: int = 10) -> dict:
    """The terms that push a linear model most towards each class (needs coef_: logistic regression or linear SVM)."""
    names = pipeline.named_steps["tfidf"].get_feature_names_out()[pipeline.named_steps["chi2"].get_support()]
    coef = pipeline.named_steps["model"].coef_
    labels = list(pipeline.named_steps["model"].classes_)
    if coef.shape[0] == 1:                                   # two classes: one weight vector
        coef = np.vstack([-coef[0], coef[0]])
    return {c: [names[j] for j in np.argsort(-coef[labels.index(c)])[:n]] for c in classes}
