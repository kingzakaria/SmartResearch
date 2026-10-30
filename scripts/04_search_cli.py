"""Step 4: search the decisions from the terminal.

Usage (from the project root):
    python scripts/04_search_cli.py "تقادم الدعوى"          one query
    python scripts/04_search_cli.py "شفعة" -k 5 --all        top 5, documents containing ALL the terms
    python scripts/04_search_cli.py --similar 8              documents closest to decision file 8
    python scripts/04_search_cli.py                          interactive mode

Interactive mode: type a query. 'sim 8' lists the documents close to 8. An empty line quits.
Words between « » in a result are the words of the text that matched the query.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402
from src.similarity.measures import MEASURES  # noqa: E402


def show_search(response, k: int) -> None:
    print(f"\nQuery: {response.query}")
    print(f"Analyzed terms: {response.terms}")
    for term in response.unknown:
        options = ", ".join(f"{t} (distance {d})" for t, d in response.suggestions.get(term, [])) or "no close term"
        print(f"  '{term}' is not in the index. Did you mean: {options}")
    if response.ignored:
        print(f"  ignored (present in every document, no discriminating power): {response.ignored}")
    if not response.results:
        print("No result.")
        return
    print(f"{response.total} document(s) found, showing the best {len(response.results)}:")
    for r in response.results:
        print(f"\n{r.rank:>2}. score {r.score:.4f}   file {r.filename}   decision {r.decision_no}   "
              f"{r.date}   case {r.case_no}")
        if r.subject:
            print(f"    subject : {r.subject}")
        print(f"    matched : {', '.join(r.matched_terms)}")
        print(f"    text    : {r.snippet}")


def show_similar(engine, doc_id: str, k: int, measure: str) -> None:
    try:
        rows = engine.similar(doc_id, k, measure)
    except KeyError as error:
        print(error)
        return
    unit = "distance, smaller = closer" if MEASURES[measure][2] else "similarity, larger = closer"
    base = engine.documents.get(str(doc_id), {})
    print(f"\nDocuments closest to {doc_id} ({measure}, {unit})")
    print(f"  {base.get('subject')}")
    for d, value in rows:
        print(f"  {value:8.4f}   {engine.documents.get(d, {}).get('filename', d):>10}   "
              f"{engine.documents.get(d, {}).get('subject')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="*", help="the query (omit for interactive mode)")
    parser.add_argument("-k", type=int, default=config.TOP_K, help="number of results")
    parser.add_argument("--all", action="store_true", help="AND: only documents containing every term")
    parser.add_argument("--similar", metavar="DOC_ID", help="list the documents closest to this one")
    parser.add_argument("--measure", default="cosine", choices=sorted(MEASURES), help="measure for --similar")
    args = parser.parse_args()

    engine = SearchEngine.from_config(config)
    if args.similar:
        show_similar(engine, args.similar, args.k, args.measure)
    elif args.query:
        show_search(engine.search(" ".join(args.query), args.k, args.all), args.k)
    else:
        print(f"{engine.index.n_docs} documents, {len(engine.index.postings)} terms. "
              "Type a query ('sim 8' for similar documents, empty line to quit).")
        while True:
            try:
                line = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            if line.startswith("sim "):
                show_similar(engine, line[4:].strip(), args.k, args.measure)
            else:
                show_search(engine.search(line, args.k, args.all), args.k)


if __name__ == "__main__":
    main()
