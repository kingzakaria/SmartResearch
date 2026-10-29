from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CLUSTERS_PATH, LABELS_PATH
from src.utils.io import load_json, save_labels

cluster_path = CLUSTERS_PATH
if not cluster_path.exists():
    raise SystemExit("Run scripts/05_cluster.py first.")
clusters = load_json(cluster_path)
labels = {doc_id: f"cluster_{cluster_id}" for doc_id, cluster_id in clusters.items()}
save_labels(LABELS_PATH, labels)
print(f"Wrote {len(labels)} provisional labels to {LABELS_PATH}")
print("Edit labels.csv to replace cluster names with meaningful categories before training.")
