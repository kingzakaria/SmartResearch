"""UTF-8 JSON, CSV, and pickle helpers."""

import csv
import json
import pickle
from pathlib import Path
from typing import Any


def save_json(path: Path | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pickle(path: Path | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        pickle.dump(value, stream)


def load_pickle(path: Path | str) -> Any:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def load_labels(path: Path | str) -> dict[str, str]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"doc_id", "category"}.issubset(reader.fieldnames):
            raise ValueError("labels CSV must have doc_id and category columns")
        return {
            row["doc_id"].strip(): row["category"].strip()
            for row in reader
            if row.get("doc_id", "").strip() and row.get("category", "").strip()
        }


def save_labels(path: Path | str, labels: dict[str, str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["doc_id", "category"])
        writer.writerows(sorted(labels.items()))
