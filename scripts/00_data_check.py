"""Step 0: sanity checks on the raw corpus. Run: python scripts/00_data_check.py"""
import hashlib
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")  # tashkeel + tatweel
LATIN = re.compile(r"[A-Za-z]")
ARABIC_LETTER = re.compile(r"[\u0621-\u064A]")
WORD = re.compile(r"[\u0621-\u064A]+")      # letters only: excludes the Arabic comma ، and digits

HEADER_CHARS = 600                           # the metadata block is at the top of each file
NUM = r"(?:رقم|عدد)"                         # both spellings are used: "رقم" and "عدد"
RE_DECISION = re.compile(NUM + r"\s*(\d{1,4}/\d{1,4})(?![\d/])")   # exactly ONE slash: 1/15
RE_DATE = re.compile(r"بتاريخ\s+(?:(\d{1,2})\s+(\S+)\s+(\d{4})|(\d{1,2}[/-]\d{1,2}[/-]\d{4}))")
RE_CASE = re.compile(r"الملف\s+(?:\S+\s+)?" + NUM + r"\s*([\d/]+)")  # 2020/1/1/4066


def read(path: Path) -> str:
    return path.read_text(encoding=config.ENCODING, errors="replace").translate(ARABIC_INDIC)


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def main() -> None:
    files = sorted(config.RAW_DIR.glob("*.txt"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
    if not files:
        print(f"No .txt files found in {config.RAW_DIR}")
        return

    texts = {f: read(f) for f in files}
    usable = {f: t for f, t in texts.items() if t.strip()}
    empty = [f.name for f, t in texts.items() if not t.strip()]
    n = len(usable)
    print(f"Files: {len(files)} | empty: {len(empty)} | usable: {n}")
    print(f"Empty files: {empty}")

    # duplicates are computed on NON-empty files only (empty files are all identical)
    groups = defaultdict(list)
    for f, t in usable.items():
        groups[hashlib.md5(t.strip().encode()).hexdigest()].append(f.name)
    dups = [g for g in groups.values() if len(g) > 1]
    print(f"Duplicate groups among non-empty files: {len(dups)}")
    for g in dups[:10]:
        print("   ", g)

    lengths = [len(WORD.findall(DIACRITICS.sub("", t))) for t in usable.values()]
    print(f"\nWords per file: mean={statistics.mean(lengths):.0f}  "
          f"median={statistics.median(lengths):.0f}  min={min(lengths)}  max={max(lengths)}")
    print(f"Total words: {sum(lengths)}")

    print(f"\nMetadata regex success (over {n} non-empty files):")
    failures = {}
    for name, rx in [("decision number", RE_DECISION), ("date", RE_DATE), ("case number", RE_CASE)]:
        in_header = [f for f, t in usable.items() if rx.search(t[:HEADER_CHARS])]
        anywhere = [f for f, t in usable.items() if rx.search(t)]
        failures[name] = [f for f in usable if f not in set(in_header)]
        print(f"  {name:16s}: in header {len(in_header)}/{n} ({pct(len(in_header), n)})"
              f"   anywhere {len(anywhere)}/{n} ({pct(len(anywhere), n)})")

    for name, bad in failures.items():
        if not bad:
            continue
        print(f"\n=== Files with NO {name} in the header: {len(bad)} ===")
        print([f.name for f in bad[:15]])
        samples = [bad[0]] + ([bad[-1]] if len(bad) > 1 else [])
        for f in samples:
            print(f"--- first 8 lines of {f.name} ---")
            print("\n".join(usable[f].strip().splitlines()[:8]))

    latin = sum(len(LATIN.findall(t)) for t in usable.values())
    arab = sum(len(ARABIC_LETTER.findall(t)) for t in usable.values())
    print(f"\nLatin letters: {latin} ({pct(latin, latin + arab)} of all letters)")

    counter = Counter()
    for t in usable.values():
        counter.update(WORD.findall(DIACRITICS.sub("", t)))
    print("\n40 most frequent words (stopword candidates):")
    print("  " + " | ".join(f"{w} ({c})" for w, c in counter.most_common(40)))

    random.seed(config.RANDOM_SEED)
    print("\n3 random files to read by eye:")
    for f in random.sample(list(usable), min(3, n)):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()