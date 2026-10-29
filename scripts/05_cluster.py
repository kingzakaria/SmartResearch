from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CLUSTER_REPORT_PATH, CLUSTERS_PATH, DEFAULT_CLUSTER_COUNT, TERMS_PATH
from src.clustering.inspect_clusters import inspect_clusters
from src.clustering.kmeans import cluster_documents
from src.utils.io import load_json, save_json

terms = load_json(TERMS_PATH)
count = min(DEFAULT_CLUSTER_COUNT, len(terms))
if count < 2:
    raise SystemExit("Clustering requires at least two analyzed documents.")
labels = cluster_documents(terms, count)
report = inspect_clusters(terms, labels)
save_json(CLUSTERS_PATH, labels)
lines = ["# Cluster inspection", ""]
for cluster, details in report.items():
    lines.extend(
        [
            f"## Cluster {cluster} ({details['size']} documents)",
            f"- Top terms: {', '.join(details['top_terms']) or '(none)'}",
            f"- Sample documents: {', '.join(details['sample_doc_ids']) or '(none)'}",
            "",
        ]
    )
CLUSTER_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Clustered {len(labels)} documents into {count} groups.")
