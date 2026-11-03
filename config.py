"""Central configuration: every path and parameter of the project lives here."""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "real_estate_chamber"   # the 365 .txt files (never modified)
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
STOPWORDS_GENERIC = RESOURCES_DIR / "stopwords_ar.txt"      # generic Arabic stopwords
STOPWORDS_LEGAL = RESOURCES_DIR / "stopwords_legal.txt"     # legal boilerplate: المحكمة، الطاعن، قرار ...

# ---------- Parameters ----------
ENCODING = "utf-8"
RANDOM_SEED = 42

ANALYZER_MODE = "light"     # "word" | "light" | "root" | "lemma"
LOG_BASE = 10               # the course example (0.0092) only works with log10

MIN_DOC_FREQ = 2            # rare zone: terms present in fewer documents than this are dropped
MAX_DOC_FREQ_RATIO = 0.8    # frequent zone: terms present in more than this share of the documents are dropped
DUPLICATE_THRESHOLD = 0.8   # same case number AND at least this Jaccard similarity: the same decision published twice
DUPLICATE_STRICT = 0.97     # without the same case number, only an almost identical text counts as a duplicate
FAMILY_THRESHOLD = 0.8      # decisions this similar share a template: kept, but never split between train and test
DUPLICATES_JSON = PROCESSED_DIR / "duplicates.json"
N_PARTITIONS = 3            # the collection is divided in this many groups (K-Means); a query searches the best one(s)
CLUSTER_MAX_DF_RATIO = 0.5  # for clustering only: terms in more than this share of the documents are dropped
PARTITIONS_JSON = MODELS_DIR / "partitions.json"
N_CLUSTERS = 6              # K-Means, to be tuned between 4 and 8
CV_FOLDS = 5                # cross-validation (small corpus)
CV_REPEATS = 3              # the cross-validation is repeated with different random splits and averaged
CLASS_MIN_SIZE = 20         # categories with fewer decisions are left out of the classification
CLASSIFIER_MIN_DF = 2       # frequency thresholding inside each training fold: terms in fewer documents are ignored
CLASSIFIER_MAX_DF = 0.5     # ... and terms in more than this share of the documents
CLASSIFIER_PKL = MODELS_DIR / "classifier.pkl"

# ---------- RAG: "Ask AI" ----------
CHUNK_SIZE = 400            # words per passage (chunk)
CHUNK_OVERLAP = 80          # words shared by two consecutive passages, so an idea cut at the border is not lost
TOP_K = 10
CHUNKS_JSON = PROCESSED_DIR / "chunks.json"
EMBEDDINGS_NPZ = MODELS_DIR / "chunk_embeddings.npz"
RAG_CANDIDATES = 20         # each retriever (keywords, embeddings) proposes this many passages
RAG_TOP_PASSAGES = 5        # passages given to the LLM after fusing both rankings
RAG_MAX_PER_DOC = 2         # at most this many passages from the same decision (variety of sources)
RRF_K = 60                  # reciprocal rank fusion constant: score = sum of 1 / (RRF_K + rank)
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_LLM_MODEL = "gemini-3.1-flash-lite"      # change here if Google renames or retires it (see AI Studio)
GEMINI_TIMEOUT = 180        # seconds to wait for Gemini before giving up on one try
GEMINI_EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768             # size of each embedding vector (the model can give 768, 1536 or 3072)
ENV_FILE = ROOT / ".env"    # holds GEMINI_API_KEY=...  (never commit it)
