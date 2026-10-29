from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import INDEX_PATH, TERMS_PATH, TFIDF_MATRIX_PATH
from src.indexing.inverted_index import InvertedIndex
from src.indexing.tfidf import TfidfIndex
from src.utils.io import load_json, save_pickle

terms = load_json(TERMS_PATH)
index = InvertedIndex(terms)
tfidf = TfidfIndex().fit(terms)
save_pickle(INDEX_PATH, index)
save_pickle(TFIDF_MATRIX_PATH, tfidf)
print(f"Indexed {len(terms)} documents, {len(index.index)} terms")
