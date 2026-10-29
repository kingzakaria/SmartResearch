"""Central configuration: every path and parameter of the project lives here."""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "real_estate_chamber"
PROCESSED_DIR = DATA_DIR / "processed"
RESOURCES_DIR = DATA_DIR / "resources"
EVAL_DIR = DATA_DIR / "eval"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# ---------- Generated files ----------
DOCUMENTS_JSON = PROCESSED_DIR / "documents.json"
TERMS_JSON = PROCESSED_DIR / "terms.json"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
INDEX_PKL = MODELS_DIR / "index.pkl"

# ---------- External resources ----------
STOPWORDS_FILES = [
    RESOURCES_DIR / "stopwords_ar.txt",
    RESOURCES_DIR / "stopwords_legal.txt",
]

# ---------- Parameters ----------
ENCODING = "utf-8"
RANDOM_SEED = 42

ANALYZER_MODE = "light"     # "word" | "light" | "root" | "lemma"
LOG_BASE = 10               # the course example (0.0092) only works with log10

MIN_DOC_FREQ = 2
N_CLUSTERS = 6
CV_FOLDS = 5

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 10