"""Pipeline A, step 1: locate and read the raw files."""
from pathlib import Path


def list_files(raw_dir: Path) -> list[Path]:
    """All .txt files, sorted numerically (8, 31, 32, ... 12839) rather than alphabetically."""
    def key(p: Path):
        return (0, int(p.stem), "") if p.stem.isdigit() else (1, 0, p.stem)
    return sorted(Path(raw_dir).glob("*.txt"), key=key)


def read_file(path: Path, encoding: str = "utf-8") -> str:
    """Read one file; undecodable bytes become U+FFFD (reported by the character audit)."""
    return Path(path).read_text(encoding=encoding, errors="replace")
