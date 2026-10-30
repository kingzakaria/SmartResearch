"""Print the first lines of the files where extraction is incomplete or suspicious.
Run from the project root: python scripts/inspect_failures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

N_LINES = 7
MAX_FILES = 5          # files shown per problem
MAX_CHARS = 110        # each line is cut to this length


def show(doc: dict) -> None:
    print(f"--- {doc['filename']} | decision={doc['decision_no']} date={doc['date']} case={doc['case_no']}")
    print(f"    subject = {doc['subject']!r}"[:MAX_CHARS + 20])
    for line in doc["text"].split("\n")[:N_LINES]:
        print("    | " + line[:MAX_CHARS])


def pick(docs: list) -> list:
    """Spread the examples over the list instead of taking only the first ones."""
    if len(docs) <= MAX_FILES:
        return docs
    stride = max(1, len(docs) // MAX_FILES)
    return docs[::stride][:MAX_FILES]


def main() -> None:
    docs = json.loads(config.DOCUMENTS_JSON.read_text(encoding="utf-8"))

    problems = {
        "NO case number": [d for d in docs if d["case_no"] is None],
        "NO decision number or NO date": [d for d in docs if d["decision_no"] is None or d["date"] is None],
        "suspicious subject (too long / too short)": [
            d for d in docs if d["subject"] and (len(d["subject"]) > 200 or len(d["subject"]) < 8)],
    }
    for title, group in problems.items():
        print(f"\n==================== {title}: {len(group)} files ====================")
        for d in pick(group):
            show(d)


if __name__ == "__main__":
    main()