"""Step 6: divide the collection into groups with K-Means (models/partitions.json).
Run from the project root:
    python scripts/07_build_partitions.py              build with N_PARTITIONS groups (config.py)
    python scripts/07_build_partitions.py --k 4        another number of groups
    python scripts/07_build_partitions.py --explore    compare k = 2..8 (cohesion and silhouette) before choosing
Needs terms.json, documents.json (steps 1-2) and duplicates.json (scripts/05_find_duplicates.py).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.clustering.kmeans import kmeans, silhouette, similarity_matrix  # noqa: E402
from src.clustering.partitioning import build_partitions, central_documents, cluster_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=config.N_PARTITIONS)
    parser.add_argument("--max-ratio", type=float, default=config.CLUSTER_MAX_DF_RATIO,
                        help="terms present in more than this share of the documents are ignored for clustering")
    parser.add_argument("--explore", action="store_true")
    args = parser.parse_args()

    terms = json.loads(config.TERMS_JSON.read_text(encoding="utf-8"))
    docs = {str(d["doc_id"]): d for d in json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))}
    chamber = config.PARTITION_CHAMBER                 # the groups are built inside ONE chamber (the supervised
    inside = {d for d, doc in docs.items() if doc.get("chamber", config.DEFAULT_CHAMBER) == chamber}   # routing handles chambers)
    terms = {d: t for d, t in terms.items() if str(d) in inside}
    print(f"Chamber {chamber}: {len(terms)} documents")
    dup = json.loads(config.DUPLICATES_JSON.read_text(encoding="utf-8")) if config.DUPLICATES_JSON.exists() else {}
    drop, groups = dup.get("drop", []), dup.get("groups", [])
    model = cluster_features(terms, drop, config.MIN_DOC_FREQ, args.max_ratio, config.LOG_BASE)
    vectors = {d: model.doc_vector(d) for d in model.index.doc_len if model.norms[d] > 0}
    print(f"{len(terms)} documents, {len(drop)} duplicates set aside, {len(vectors)} clustered; "
          f"clustering vocabulary: {len(model.index.postings)} terms (upper limit {args.max_ratio:.0%}).\n")

    if args.explore:
        ids, sims = similarity_matrix(vectors)
        print(f"{'k':>3}{'cohesion':>10}{'silhouette':>12}   group sizes")
        for k in range(2, 9):
            assign, _, cohesion = kmeans(vectors, k, seed=config.RANDOM_SEED, n_init=5)
            sizes = sorted((sum(1 for c in assign.values() if c == i) for i in range(k)), reverse=True)
            print(f"{k:>3}{cohesion:>10.3f}{silhouette(ids, sims, assign):>12.3f}   {sizes}")
        print("\nHigher silhouette = better separated groups. Do not expect high values on legal text.")
        return

    data = build_partitions(terms, args.k, drop, groups, config.RANDOM_SEED, config.MIN_DOC_FREQ,
                            args.max_ratio, config.LOG_BASE)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data["chamber"] = chamber
    config.PARTITIONS_JSON.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    centrals = central_documents(data, {d: model.doc_vector(d) for d in vectors})
    print(f"K-Means, k={args.k}: mean cosine to the centroid = {data['cohesion']}  (sizes incl. duplicates: {data['sizes']})\n")
    for c in range(args.k):
        print(f"=== Group {c + 1}: {data['sizes'][c]} decisions")
        print("  main terms : " + " | ".join(data["top_terms"][c]))
        for d in centrals[c]:
            print(f"  decision {docs[d].get('decision_no')}: {docs[d].get('subject')}")
        print()

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "partitions.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "group", "decision_no", "case_no", "subject"])
        for d, g in sorted(data["assignments"].items(), key=lambda kv: (kv[1], int(kv[0]) if kv[0].isdigit() else 0)):
            writer.writerow([d, g + 1, docs[d].get("decision_no"), docs[d].get("case_no"), docs[d].get("subject")])
    print(f"Saved: {config.PARTITIONS_JSON}  and  {config.REPORTS_DIR / 'partitions.csv'}")
    print("Read reports/partitions.csv in Excel: do the subjects of a group share a topic? Then name the groups.")


if __name__ == "__main__":
    main()
