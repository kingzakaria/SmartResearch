"""Predict a category from analyzed terms."""

from sklearn.pipeline import Pipeline


def predict_category(model: Pipeline, terms: list[str]) -> str:
    if not terms:
        raise ValueError("cannot classify an empty document")
    return str(model.predict([" ".join(terms)])[0])
