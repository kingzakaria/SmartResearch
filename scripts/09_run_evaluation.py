from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    ARABIC_STOPWORDS_PATH,
    DEFAULT_TOP_K,
    DOCUMENTS_PATH,
    LEGAL_STOPWORDS_PATH,
    QUERIES_PATH,
    SEARCH_METRICS_PATH,
)
from src.evaluation.search_metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from src.morphology.analyzer import ArabicMorphologyAnalyzer
from src.search.engine import SearchEngine
from src.utils.io import load_json

documents = load_json(DOCUMENTS_PATH)
queries = load_json(QUERIES_PATH)
if not documents:
    raise SystemExit("No documents available. Run scripts/01_extract.py first.")
analyzer = ArabicMorphologyAnalyzer([ARABIC_STOPWORDS_PATH, LEGAL_STOPWORDS_PATH])
engine = SearchEngine(documents, analyzer)
results = []
for item in queries:
    relevant = set(item.get("relevant_doc_ids", []))
    if not relevant:
        continue
    ranked = [result["doc_id"] for result in engine.search(item["query"], top_k=DEFAULT_TOP_K)]
    results.append(
        {
            "query": item["query"],
            f"precision@{DEFAULT_TOP_K}": precision_at_k(ranked, relevant, DEFAULT_TOP_K),
            f"recall@{DEFAULT_TOP_K}": recall_at_k(ranked, relevant, DEFAULT_TOP_K),
            "average_precision": average_precision(ranked, relevant),
            f"ndcg@{DEFAULT_TOP_K}": ndcg_at_k(ranked, relevant, DEFAULT_TOP_K),
        }
    )
SEARCH_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
target = SEARCH_METRICS_PATH
with target.open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "query",
            f"precision@{DEFAULT_TOP_K}",
            f"recall@{DEFAULT_TOP_K}",
            "average_precision",
            f"ndcg@{DEFAULT_TOP_K}",
        ],
    )
    writer.writeheader()
    writer.writerows(results)
print(f"Evaluated {len(results)} queries with relevance judgments; report: {target}")
if not results:
    print(f"Add relevant_doc_ids to {QUERIES_PATH} to calculate search metrics.")
