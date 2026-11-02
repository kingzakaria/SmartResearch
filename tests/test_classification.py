"""Classification dataset and cross-validation. Run: pytest tests/test_classification.py"""
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.classification.dataset import Dataset, build_dataset, classification_text, subject_leak_rate  # noqa: E402
from src.classification.evaluate import (cross_validate, folds_keep_groups_together, make_pipeline,  # noqa: E402
                                         top_terms_per_class)
from src.morphology.analyzer import analyzer_from_config  # noqa: E402

ANALYZER = analyzer_from_config(config, "light")
TOPICS = {"a": ["شفع", "بيع", "عقار", "ثمن"], "b": ["تحفيظ", "تعرض", "مطلب", "رسم"], "c": ["افراغ", "كراء", "احتلال", "منزل"]}


def synthetic(n_per_class=20, seed=0) -> Dataset:
    rng = random.Random(seed)
    common = [f"w{i}" for i in range(30)]
    ids, tokens, y, groups = [], [], [], []
    for label, words in TOPICS.items():
        for i in range(n_per_class):
            tokens.append([rng.choice(words) for _ in range(12)] + [rng.choice(common) for _ in range(20)])
            ids.append(f"{label}{i}"); y.append(label)
            groups.append(f"family-{label}" if i < 4 else f"doc-{label}{i}")      # 4 template twins per class
    return Dataset(ids, tokens, y, groups, sorted(TOPICS))


def test_cross_validation_learns_the_topics_and_beats_the_baseline():
    ds = synthetic()
    base = cross_validate(ds, "majority", repeats=1)
    for model in ("naive_bayes", "rocchio", "knn", "logistic_regression", "linear_svm"):
        r = cross_validate(ds, model, percentile=50, min_df=2, max_df=0.9, repeats=1)
        assert r["accuracy"] > 0.9 and r["macro_f1"] > base["macro_f1"] + 0.3, model
    assert len(r["pooled"]) == len(ds.y)


def test_groups_are_never_split_between_folds():
    assert folds_keep_groups_together(synthetic())


def test_top_terms_of_a_linear_model_are_the_topic_words():
    ds = synthetic()
    pipe = make_pipeline("logistic_regression", 50, 2, 0.9).fit(ds.tokens, ds.y)
    top = top_terms_per_class(pipe, ds.classes, 3)
    assert set(top["b"]) <= set(TOPICS["b"]) and set(top["c"]) <= set(TOPICS["c"])


DOCS = [
    {"doc_id": 1, "subject": "نزاع تحفيظ - عبء الإثبات", "body": "نزاع تحفيظ - عبء الإثبات\nحيث إن الطاعن يعيب على القرار"},
    {"doc_id": 2, "subject": "دعوى الإفراغ - أثره", "body": "دعوى الإفراغ - أثره\nوحيث إن المحكمة ثبت لها الاحتلال"},
    {"doc_id": 3, "subject": "دعوى الإفراغ - أثره", "body": "نفس القرار"},
    {"doc_id": 4, "subject": "شفعة", "body": "شفعة\nنص"},
    {"doc_id": 5, "subject": None, "body": "بلا موضوع"},
]
LABELS = {"1": "registration", "2": "eviction", "3": "eviction", "4": "preemption", "5": "other"}


def test_build_dataset_leaves_out_duplicates_other_and_small_classes():
    ds = build_dataset(DOCS, LABELS, drop=["3"], families=[["1", "2"]], analyzer=ANALYZER, min_class_size=1)
    assert ds.ids == ["1", "2", "4"] and ds.left_out == {"duplicate": 1, "no category": 1}
    assert ds.groups[0] == ds.groups[1] and ds.groups[2] == "doc4"
    small = build_dataset(DOCS, LABELS, drop=["3"], analyzer=ANALYZER, min_class_size=2)
    assert small.ids == [] and small.left_out["class 'registration' too small"] == 1


def test_the_subject_line_is_not_in_the_classification_text():
    assert "تحفيظ" not in classification_text(DOCS[0]) and "الطاعن" in classification_text(DOCS[0])
    rate, leaking = subject_leak_rate(DOCS, ["1", "2"])
    assert rate == 0 and leaking == []
    leaky = [{"doc_id": 9, "subject": "نزاع تحفيظ - عبء الإثبات", "body": "حيث إن نزاع تحفيظ - عبء الإثبات وارد هنا"}]
    assert subject_leak_rate(leaky, ["9"])[0] == 1.0
