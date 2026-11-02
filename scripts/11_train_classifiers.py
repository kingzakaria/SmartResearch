"""Step 7: classify the decisions into categories, with cross-validation.
Run from the project root:
    python scripts/11_train_classifiers.py             full comparison (a minute or two)
    python scripts/11_train_classifiers.py --quick     one repeat of the cross-validation
Needs: data/processed/documents.json, duplicates.json, labels.csv (scripts 01, 05, 10) and scikit-learn.
Writes reports/classification_results.csv, reports/confusion_matrix.csv and models/classifier.pkl.
"""
import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    sys.exit("scikit-learn is missing. Run:  pip install scikit-learn")

from src.classification.dataset import build_dataset, subject_leak_rate  # noqa: E402
from src.classification.evaluate import cross_validate, make_pipeline, top_terms_per_class  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402

PERCENTILES = (10, 25, 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    repeats = 1 if parser.parse_args().quick else config.CV_REPEATS

    docs = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))
    dup = json.loads(config.DUPLICATES_JSON.read_text(encoding="utf-8")) if config.DUPLICATES_JSON.exists() else {}
    with open(config.LABELS_CSV, encoding="utf-8-sig", newline="") as f:
        labels = {row["doc_id"]: row["category"] for row in csv.DictReader(f)}
    analyzer = analyzer_from_config(config)
    ds = build_dataset(docs, labels, dup.get("drop", []), dup.get("families", []), analyzer, config.CLASS_MIN_SIZE)

    sizes = Counter(ds.y)
    print(f"{len(ds.ids)} decisions, {len(ds.classes)} categories, {len(set(ds.groups))} groups for the folds "
          f"(duplicates and template families stay together)")
    for c in ds.classes:
        print(f"   {c:<22}{sizes[c]:>5}  {sizes[c] / len(ds.y):>6.1%}")
    print(f"Left out: {ds.left_out}")
    rate, leaking = subject_leak_rate(docs, ds.ids)
    print(f"Leak check: the subject line still appears in the text of {len(leaking)} decisions ({rate:.1%})"
          + ("  <-- investigate: " + ", ".join(leaking[:8]) if rate > 0.03 else "  (fine)"))

    base = cross_validate(ds, "majority", repeats=1)
    print(f"\nBaseline (always the biggest category): accuracy {base['accuracy']:.1%}, macro-F1 {base['macro_f1']:.3f}")
    print("A classifier must clearly beat this. Accuracy alone is misleading with unbalanced categories: read macro-F1.\n")

    results = []
    print(f"{'model':<22}{'% terms kept (chi2)':>20}{'accuracy':>16}{'macro-F1':>16}")
    for model in ("naive_bayes", "rocchio", "knn", "logistic_regression", "linear_svm"):
        for percentile in PERCENTILES:
            r = cross_validate(ds, model, percentile, config.CLASSIFIER_MIN_DF, config.CLASSIFIER_MAX_DF,
                               config.CV_FOLDS, repeats, config.RANDOM_SEED)
            results.append(r)
            print(f"{model:<22}{percentile:>19}%{r['accuracy']:>9.1%} ±{r['accuracy_std']:.3f}{r['macro_f1']:>9.3f} ±{r['macro_f1_std']:.3f}")
    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\nBest: {best['model']} keeping {best['percentile']}% of the terms: accuracy {best['accuracy']:.1%}, "
          f"macro-F1 {best['macro_f1']:.3f}")

    print("\nEffect of the upper frequency limit (max_df), logistic regression:")
    for max_df in (0.3, 0.5, 0.8, 1.0):
        r = cross_validate(ds, "logistic_regression", best["percentile"], config.CLASSIFIER_MIN_DF, max_df,
                           config.CV_FOLDS, repeats, config.RANDOM_SEED)
        print(f"   terms in more than {max_df:.0%} of the decisions ignored: accuracy {r['accuracy']:.1%}, macro-F1 {r['macro_f1']:.3f}")

    print(f"\nPer category ({best['model']}, each decision predicted by a model that never saw it):")
    print(classification_report(ds.y, best["pooled"], labels=ds.classes, zero_division=0, digits=2))
    matrix = confusion_matrix(ds.y, best["pooled"], labels=ds.classes)
    print("Confusion matrix (rows = true category, columns = predicted):")
    print(" " * 22 + "".join(f"{c[:9]:>11}" for c in ds.classes))
    for c, row in zip(ds.classes, matrix):
        print(f"{c:<22}" + "".join(f"{v:>11}" for v in row))

    # final model on all the data + the words that characterise each category
    final = make_pipeline(best["model"], best["percentile"], config.CLASSIFIER_MIN_DF, config.CLASSIFIER_MAX_DF)
    final.fit(ds.tokens, ds.y)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CLASSIFIER_PKL, "wb") as f:
        pickle.dump({"pipeline": final, "classes": ds.classes, "mode": config.ANALYZER_MODE,
                     "cv": {k: best[k] for k in ("model", "percentile", "accuracy", "macro_f1")}}, f)
    linear = make_pipeline("logistic_regression", best["percentile"], config.CLASSIFIER_MIN_DF, config.CLASSIFIER_MAX_DF)
    linear.fit(ds.tokens, ds.y)
    print("\nWords that push a decision towards each category (logistic regression). Do they make legal sense?")
    for c, words in top_terms_per_class(linear, ds.classes, 10).items():
        print(f"   {c:<22}" + " | ".join(words))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "classification_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "percent_terms_kept", "accuracy", "accuracy_std", "macro_f1", "macro_f1_std"])
        for r in results:
            writer.writerow([r["model"], r["percentile"], f"{r['accuracy']:.4f}", f"{r['accuracy_std']:.4f}",
                             f"{r['macro_f1']:.4f}", f"{r['macro_f1_std']:.4f}"])
    with open(config.REPORTS_DIR / "confusion_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ predicted"] + ds.classes)
        for c, row in zip(ds.classes, matrix):
            writer.writerow([c] + list(row))
    print(f"\nSaved: {config.CLASSIFIER_PKL}, reports/classification_results.csv, reports/confusion_matrix.csv")
    print("Remember: the labels come from keyword rules on the subject line, so these scores measure how well the text\n"
          "predicts that labelling, not an absolute truth.")


if __name__ == "__main__":
    main()
