from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CLASSIFICATION_METRICS_PATH, CLASSIFIER_PATH, LABELS_PATH, TERMS_PATH
from src.classification.evaluate import evaluate_classifier
from src.classification.train import train_classifier
from src.utils.io import load_json, load_labels, save_json, save_pickle

terms = load_json(TERMS_PATH)
labels = load_labels(LABELS_PATH)
model = train_classifier(terms, labels)
metrics = evaluate_classifier(model, terms, labels)
save_pickle(CLASSIFIER_PATH, model)
save_json(CLASSIFICATION_METRICS_PATH, metrics)
print(f"Saved classifier to {CLASSIFIER_PATH}")
print(f"Accuracy: {metrics['accuracy']:.3f}; macro F1: {metrics['macro_f1']:.3f}")
