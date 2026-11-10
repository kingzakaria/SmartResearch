"""Naive Bayes chamber classifier (the TP of the course), used to route a search query.
Run from the project root: python scripts/15_train_naive_bayes.py

1. TP evaluation: the last 20 decisions of each chamber are the test set, the others train the model.
   Whole-document accuracy and confusion matrix, as in the TP.
2. QUERY evaluation (what the search really needs): 5-fold cross-validation where the test "queries" are the subject
   line of each held-out decision (long query) and 3 words drawn from it (short query, closer to what a user types).
   Reports the accuracy of the prediction, how often the model is sure enough (probability >= the confidence
   threshold of config.py), and how accurate it is when it is sure.
3. The final model is trained on every non-duplicate decision and saved to models/naive_bayes.json.

Needs data/processed/{documents,terms,duplicates}.json (scripts 01, 02, 05) and decisions of at least two chambers.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.classification.naive_bayes import NaiveBayes  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402


def file_order(doc):
    stem = str(doc["doc_id"]).split("-")[-1]
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)


def confusion(truth, predicted, classes):
    table = {t: Counter() for t in classes}
    for t, p in zip(truth, predicted):
        table[t][p] += 1
    return table


def print_confusion(table, classes):
    width = max(len(c) for c in classes) + 2
    print(f"{'true \\ predicted':<{width + 2}}" + "".join(f"{c:>{width}}" for c in classes))
    for t in classes:
        print(f"{t:<{width + 2}}" + "".join(f"{table[t][p]:>{width}}" for p in classes))


def main() -> None:
    docs = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))
    terms = json.loads(config.TERMS_JSON.read_text(encoding="utf-8"))
    dup = json.loads(config.DUPLICATES_JSON.read_text(encoding="utf-8")) if config.DUPLICATES_JSON.exists() else {}
    drop = {str(d) for d in dup.get("drop", [])}
    analyzer = analyzer_from_config(config)

    rows = [(d, d.get("chamber", config.DEFAULT_CHAMBER)) for d in docs
            if str(d["doc_id"]) not in drop and terms.get(str(d["doc_id"]))]
    by_chamber = {}
    for d, c in rows:
        by_chamber.setdefault(c, []).append(d)
    classes = sorted(by_chamber)
    if len(classes) < 2:
        sys.exit(f"Only one chamber found ({classes}). Put the decisions of the second chamber in "
                 f"{config.PERSONAL_STATUS_DIR} and run scripts 01, 02, 03 and 05 again.")
    print("Decisions per chamber (duplicates set aside): " + ", ".join(f"{c}: {len(by_chamber[c])}" for c in classes))

    # ---------- 1. the TP split: last N of each chamber are the test ----------
    train, test = [], []
    for c in classes:
        ordered = sorted(by_chamber[c], key=file_order)
        n_test = min(config.NB_TEST_PER_CLASS, max(1, len(ordered) // 5))
        train += [(d, c) for d in ordered[:-n_test]]
        test += [(d, c) for d in ordered[-n_test:]]
    model = NaiveBayes().fit([terms[str(d["doc_id"])] for d, _ in train], [c for _, c in train])
    print(f"\n=== 1. TP split: {len(train)} train / {len(test)} test; vocabulary m = {model.m}")
    for c in classes:
        print(f"  P({c}) = {model.n_docs[c]}/{len(train)} = {model.prior(c):.5f}   N_c = {model.totals[c]}")
    predicted = [model.predict(terms[str(d["doc_id"])]) for d, _ in test]
    truth = [c for _, c in test]
    print(f"  accuracy on the {len(test)} test decisions: {sum(p == t for p, t in zip(predicted, truth)) / len(test):.1%}")
    print_confusion(confusion(truth, predicted, classes), classes)

    # ---------- 2. query-level evaluation ----------
    rng = random.Random(config.RANDOM_SEED)
    folds = {c: [] for c in classes}
    for c in classes:
        ids = [str(d["doc_id"]) for d in by_chamber[c]]
        rng.shuffle(ids)
        folds[c] = [ids[i::5] for i in range(5)]
    docs_by_id = {str(d["doc_id"]): d for d, _ in rows}
    chamber_of = {str(d["doc_id"]): c for d, c in rows}
    items = {"subject (long query)": [], "3 words of the subject (short query)": []}
    for f in range(5):
        held = {i for c in classes for i in folds[c][f]}
        fit_ids = [i for i in docs_by_id if i not in held]
        cv = NaiveBayes().fit([terms[i] for i in fit_ids], [chamber_of[i] for i in fit_ids])
        for i in held:
            subject = docs_by_id[i].get("subject")
            q_terms = analyzer.analyze(subject) if subject else []
            if not q_terms:
                continue
            items["subject (long query)"].append((chamber_of[i], cv, q_terms))
            items["3 words of the subject (short query)"].append((chamber_of[i], cv, rng.sample(q_terms, min(3, len(q_terms)))))
    print(f"\n=== 2. Queries built from held-out decisions (5-fold CV), confidence threshold {config.CHAMBER_MIN_CONFIDENCE:.0%}")
    print(f"{'query':<38}{'priors':<11}{'n':>5}{'accuracy':>10}{'sure':>8}{'acc. if sure':>14}{'wrong & sure':>14}")
    sweep = {"empirical": [], "uniform": []}              # (correct, best probability, known words) of every query
    for name, rows_q in items.items():
        for priors in ("empirical", "uniform"):
            out = []
            for truth_c, cv, q_terms in rows_q:
                proba = cv.predict_proba(q_terms, priors == "uniform")
                sweep[priors].append((max(proba, key=proba.get) == truth_c, max(proba.values()), cv.n_known(q_terms)))
                out.append((max(proba, key=proba.get) == truth_c, max(proba.values()) >= config.CHAMBER_MIN_CONFIDENCE and cv.n_known(q_terms) > 0))
            n, n_sure = len(out), sum(s for _, s in out)
            if not n:
                print(f"{name:<38}{priors:<11}  no query (no subject found in the decisions)")
                continue
            acc_sure = sum(o for o, s in out if s) / n_sure if n_sure else 0
            print(f"{name:<38}{priors:<11}{n:>5}{sum(o for o, _ in out) / n:>10.1%}{n_sure / n:>8.1%}{acc_sure:>14.1%}{sum(1 for o, s in out if s and not o):>14}")
    print("\n  Choosing the confidence threshold (both kinds of query together): a higher threshold sends fewer queries to the")
    print("  wrong chamber but narrows fewer queries.")
    print(f"  {'priors':<11}{'threshold':>10}{'narrowed':>10}{'acc. if sure':>14}{'wrong & sure (% of all queries)':>34}")
    for priors, rows_p in sweep.items():
        for threshold in (0.6, 0.8, 0.9, 0.95, 0.99):
            sure = [(ok, top) for ok, top, known in rows_p if top >= threshold and known > 0]
            if not rows_p or not sure:
                continue
            wrong = sum(1 for ok, _ in sure if not ok)
            print(f"  {priors:<11}{threshold:>10.2f}{len(sure) / len(rows_p):>10.1%}{sum(ok for ok, _ in sure) / len(sure):>14.1%}{wrong / len(rows_p):>34.1%}")
    print("  priors: 'empirical' = P(c) from the number of decisions (the TP); 'uniform' = every chamber equally likely first.")
    print(f"  config.NB_PRIORS = {config.NB_PRIORS!r} is the one the search uses: pick the row with fewer 'wrong & sure'.")
    print("  'sure' = the best probability reaches the threshold: only then is a single chamber searched.")
    print("  'wrong & sure' = queries sent to the wrong chamber: the only case where relevant decisions are missed.")

    # ---------- 3. final model ----------
    final = NaiveBayes().fit([terms[str(d["doc_id"])] for d, _ in rows], [c for _, c in rows])
    meta = {"mode": analyzer.mode, "documents": len(rows), "chambers": classes, "vocabulary": final.m}
    final.save(config.NB_MODEL_JSON, meta)
    print(f"\nFinal model trained on {len(rows)} decisions, saved to {config.NB_MODEL_JSON}")
    print("Restart the API to use it: the search page then has the 'Auto' chamber option working.")


if __name__ == "__main__":
    main()
