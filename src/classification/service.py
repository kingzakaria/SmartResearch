"""Predict the category of a text with the model saved by scripts/11_train_classifiers.py."""
import pickle

import numpy as np

from ..extraction.cleaner import strip_panel_names
from .dataset import classification_tokens


class Classifier:
    def __init__(self, pipeline, classes, analyzer, info: dict | None = None):
        self.pipeline, self.classes, self.analyzer, self.info = pipeline, list(classes), analyzer, info or {}

    @classmethod
    def from_config(cls, config, analyzer) -> "Classifier":
        with open(config.CLASSIFIER_PKL, "rb") as f:
            data = pickle.load(f)
        if data.get("mode") != analyzer.mode:
            raise ValueError(f"classifier.pkl was trained with mode {data.get('mode')!r}, config uses {analyzer.mode!r}: "
                             "run scripts/11_train_classifiers.py again")
        return cls(data["pipeline"], data["classes"], analyzer, data.get("cv"))

    def predict(self, text: str) -> list[tuple[str, float]]:
        """[(category, probability)] from the most to the least likely. Empty if the text has no usable word."""
        tokens = classification_tokens(strip_panel_names(text), self.analyzer)
        if not tokens:
            return []
        model = self.pipeline.named_steps["model"]
        if hasattr(model, "predict_proba"):
            scores = self.pipeline.predict_proba([tokens])[0]
        else:                                         # linear SVM, Rocchio: turn the margins into a distribution
            margins = np.atleast_2d(self.pipeline.decision_function([tokens]))[0]
            exp = np.exp(margins - margins.max())
            scores = exp / exp.sum()
        labels = list(model.classes_)
        return sorted(((str(c), float(s)) for c, s in zip(labels, scores)), key=lambda cs: -cs[1])
