"""Reading the relevance judgments that a person fills in a spreadsheet (reports/eval_pool.csv).

Excel in some languages saves a CSV with ';' instead of ',' and, unless "CSV UTF-8" is chosen, in a legacy encoding that
destroys Arabic: both cases are handled or reported.
"""
import csv
import io
from pathlib import Path

YES = {"1", "y", "yes", "oui", "true", "نعم", "x"}
NO = {"0", "n", "no", "non", "false", "لا"}


def parse_relevance(value) -> int | None:
    """1, 0, or None when the cell is empty (not judged yet). Anything else is an error."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in YES:
        return 1
    if text in NO:
        return 0
    raise ValueError(f"cannot read the judgment {value!r}: write 1 (relevant) or 0 (not relevant)")


def read_text(path: Path) -> str:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1256"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"{path} is not in a readable encoding: in Excel use 'Save as > CSV UTF-8'")
    if "�" in text or ("?" in text and not any("؀" <= c <= "ۿ" for c in text)):
        raise ValueError(f"the Arabic text of {path} was damaged when saving: in Excel use 'Save as > CSV UTF-8'")
    return text


def read_pool(path: Path) -> list[dict]:
    """The rows of the pool file with the judgment parsed ('relevant' becomes 1, 0 or None)."""
    text = read_text(path)
    first = text.splitlines()[0] if text else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    for row in rows:
        row["relevant"] = parse_relevance(row.get("relevant"))
    return rows


def read_compact(path: Path) -> dict[tuple[str, str], int]:
    """data/eval/judgments.csv (query_id, doc_id, relevant), the small file that is kept in the repository."""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {(r["query_id"], r["doc_id"]): int(r["relevant"]) for r in csv.DictReader(fh)}


def write_compact(path: Path, judged: dict[tuple[str, str], int]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_id", "doc_id", "relevant"])
        for (query_id, doc_id), value in sorted(judged.items()):
            writer.writerow([query_id, doc_id, value])
