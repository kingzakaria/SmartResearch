"""Try the RAG from the terminal. python scripts/14_ask_cli.py "ما شروط الشفعة؟" [--no-llm]
--no-llm only shows the retrieved passages (no API key needed for the keyword part)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.rag.gemini import GeminiError  # noqa: E402
from src.rag.store import build_service  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402


class NoLLM:
    def embed_query(self, text):
        raise RuntimeError("not used with --no-llm")

    def generate(self, *a):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    engine = SearchEngine.from_config(config)
    service = build_service(config, engine, NoLLM() if args.no_llm else None)
    if args.no_llm:
        for p in service.retriever.retrieve(args.question, config.RAG_TOP_PASSAGES):
            d = engine.documents[p.doc_id]
            print(f"[{p.n}] decision {d.get('decision_no')}  keyword rank {p.keyword_rank}  score {p.score:.4f}")
            print("    " + p.text[:200].replace("\n", " ") + "...\n")
        return
    try:
        result = service.ask(args.question)
    except GeminiError as error:
        sys.exit(f"Gemini error ({error.status}): {error}\nIf the model is retired, change GEMINI_LLM_MODEL in config.py.")
    print(result["answer"])
    print(f"\nretrieval: {result['retrieval']}   {result['warning'] or ''}")
    for s in result["sources"]:
        print(f"  [{s['n']}] {'cited ' if s['cited'] else '       '} decision {s['decision_no']}  ({s['date']})")


if __name__ == "__main__":
    main()
