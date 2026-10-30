"""Show what the morphology pipeline does to a sentence, token by token, in the three modes.
Run: python scripts/demo_morphology.py ["your sentence"]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402

DEFAULT = ("وان المحكمة لما اعتبرت أن الطاعنين تمسكوا بالشفعة بمقتضى الفصل 371 "
           "من قانون الالتزامات والعقود فإن قرارها المطعون فيه يكون معللا")


def main() -> None:
    text = " ".join(sys.argv[1:]) or DEFAULT
    print(f"Text: {text}\n")
    analyzers = {m: analyzer_from_config(config, m) for m in ("word", "light", "root")}
    rows = {m: a.explain(text) for m, a in analyzers.items()}
    print(f"{'token':16s}{'status':13s}{'light':12s}{'root':12s}")
    for (tok, status, light), (_, _, root) in zip(rows["light"], rows["root"]):
        print(f"{tok:16s}{status:13s}{light:12s}{root:12s}")
    for m, a in analyzers.items():
        print(f"\n[{m}] {a.analyze(text)}")


if __name__ == "__main__":
    main()
