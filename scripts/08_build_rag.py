from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DOCUMENTS_PATH, VECTOR_STORE_PATH
from src.rag.chunker import chunk_documents
from src.rag.vector_store import VectorStore
from src.utils.io import load_json

documents = load_json(DOCUMENTS_PATH)
chunks = chunk_documents(documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
if not chunks:
    raise SystemExit("No text chunks to index. Add source files and run scripts/01_extract.py.")
VectorStore(chunks).save(VECTOR_STORE_PATH)
print(f"Indexed {len(chunks)} passages from {len(documents)} documents.")
