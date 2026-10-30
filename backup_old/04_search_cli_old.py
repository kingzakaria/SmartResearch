from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ARABIC_STOPWORDS_PATH, DEFAULT_TOP_K, DOCUMENTS_PATH, LEGAL_STOPWORDS_PATH
from src.morphology.analyzer import ArabicMorphologyAnalyzer
from src.search.engine import SearchEngine
from src.utils.io import load_json

documents = load_json(DOCUMENTS_PATH)
if not documents:
    raise SystemExit("No documents found. Run scripts/01_extract.py after adding source files.")
engine = SearchEngine(
    documents, ArabicMorphologyAnalyzer([ARABIC_STOPWORDS_PATH, LEGAL_STOPWORDS_PATH])
)
print("Enter an Arabic search query; type 'q' to quit.")
while True:
    query = input("> ").strip()
    if query.casefold() in {"q", "quit", "exit"}:
        break
    for result in engine.search(query, DEFAULT_TOP_K):
        metadata = result["document"].get("metadata", {})
        print(f"\n{result['doc_id']}  score={result['score']:.4f}  "
              f"decision={metadata.get('decision_number')}")
        print(str(result["document"]["text"])[:400].replace("\n", " "))
