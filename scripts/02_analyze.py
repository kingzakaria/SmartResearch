from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ARABIC_STOPWORDS_PATH, DOCUMENTS_PATH, LEGAL_STOPWORDS_PATH, TERMS_PATH
from src.morphology.analyzer import ArabicMorphologyAnalyzer
from src.utils.io import load_json, save_json

documents = load_json(DOCUMENTS_PATH)
analyzer = ArabicMorphologyAnalyzer([ARABIC_STOPWORDS_PATH, LEGAL_STOPWORDS_PATH])
terms = {document["doc_id"]: analyzer.analyze(document["text"]) for document in documents}
save_json(TERMS_PATH, terms)
print(f"Analyzed {len(terms)} documents to {TERMS_PATH}")
