"""Step 6b: full search against partitioned search: how much work is saved, and how much accuracy is lost?
Run from the project root: python scripts/08_compare_partitioned.py
Test queries: the subject of every decision (a realistic short legal query) of the chamber the groups were built for.
The groups exist for ONE chamber (config.PARTITION_CHAMBER): both searches below stay inside it.
Writes reports/partition_comparison.csv.
"""
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402

TOP = 10


def timed(function, repeat: int = 5) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best * 1000


def main() -> None:
    engine = SearchEngine.from_config(config)
    if engine.partitions is None:
        sys.exit("No partitions: run python scripts/07_build_partitions.py first.")
    p, k = engine.partitions, engine.partitions.k
    chamber = engine.partitions_chamber
    inside = {d for d, c in engine.chambers.items() if c == chamber}
    queries = sorted({doc["subject"] for d, doc in engine.documents.items() if d in inside and doc.get("subject")})
    print(f"Chamber {chamber}: {len(queries)} queries, {len(inside)} documents, {k} groups, top {TOP}\n")

    rows, summary = [], []
    for n_probe in range(1, k + 1):
        overlap, top1, scored, t_full, t_part, used = [], [], [], [], [], 0
        for q in queries:
            terms = engine.analyzer.analyze(q)
            full = engine.search(q, k=TOP, chamber=chamber)
            part = engine.search(q, k=TOP, n_probe=n_probe, chamber=chamber)
            if not full.results:
                continue
            used += 1
            a, b = [r.doc_id for r in full.results], [r.doc_id for r in part.results]
            overlap.append(len(set(a) & set(b)) / len(a))
            top1.append(a[0] == (b[0] if b else None))
            scored.append(part.scored / max(1, full.scored))
            t_full.append(timed(lambda: engine.model.cosine_scores(terms, inside), 3))
            groups, _, _ = p.route(terms, n_probe)
            t_part.append(timed(lambda: (p.route(terms, n_probe), p.cosine_scores(terms, groups)), 3))
            rows.append([n_probe, q[:60], f"{overlap[-1]:.2f}", a[0] == (b[0] if b else None), full.scored, part.scored])
        summary.append((n_probe, statistics.mean(overlap), statistics.mean(top1), statistics.mean(scored),
                        statistics.mean(t_full), statistics.mean(t_part), used))

    print(f"{'groups searched':>16}{'top-10 kept':>13}{'top-1 kept':>12}{'docs scored':>13}{'time full':>11}{'time part.':>12}")
    for n_probe, ov, t1, sc, tf_, tp, used in summary:
        print(f"{n_probe:>12} / {k}{ov:>13.1%}{t1:>12.1%}{sc:>13.1%}{tf_:>9.2f}ms{tp:>10.2f}ms")
    print(f"\n(measured on {summary[0][6]} queries that return results; 'docs scored' is relative to the full search.)")
    print("Reading: 'top-10 kept' = share of the full search's top 10 also found by the partitioned search.")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "partition_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["n_probe", "query", "top10_overlap", "top1_same", "scored_full", "scored_partitioned"])
        writer.writerows(rows)
    print(f"Saved: {config.REPORTS_DIR / 'partition_comparison.csv'}")


if __name__ == "__main__":
    main()
