# Arabic Legal Decisions Search

A reproducible, offline-first text-mining project for Arabic legal decisions. It
includes Arabic document extraction and analysis, a web-based search interface,
TF-IDF/BM25 retrieval, decision similarity, clustering, supervised
classification, a retrieval-augmented generation (RAG) pipeline, and evaluation
utilities. The web interface is in Arabic and uses a right-to-left layout.

## Requirements

- Python 3.10 or later
- Node.js and npm (for the React web interface)

## Set up the Python environment

From the repository root:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Python requirements include the FastAPI service used by the web interface.

## Prepare the decision index

Put UTF-8 `.txt` decisions in `data/raw/real_estate_chamber/`. The project does
not include private case files. Keep source files unchanged; extracted and
processed data are written to `data/processed/`, and the search index is written
to `models/`.

Run the indexing steps from the repository root:
```powershell
python scripts/00_data_check.py
python scripts/01_extract.py
python scripts/02_analyze.py
python scripts/03_build_index.py
```

The API loads `models/index.pkl` when it first needs the index. Build the index
before starting the web interface if that file is not already present.

## Run the web interface

Start the API in one terminal:
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.api:app --reload --port 8000
```

In a second terminal, install the frontend dependencies and start the React app:
```powershell
cd frontend
npm ci
npm run dev
```

Open the local address printed by Vite (by default
`http://localhost:5173`). During development, Vite forwards `/api` requests to
the API on port 8000. The API's interactive documentation is available at
`http://localhost:8000/docs`.

The interface supports Arabic query search, optional matching of all query
words, suggested terms, paginated results, highlighted snippets, and full
decision text with similar-decision links. Similarity can be explored using
cosine, Dice, Jaccard, Euclidean, and Manhattan measures.

## Run the pipeline

After preparing the index, run other pipeline steps from the repository root
as needed:

```powershell
python scripts/04_search_cli.py
python scripts/05_cluster.py
python scripts/06_label.py
python scripts/07_train_classifier.py
python scripts/08_build_rag.py
python scripts/09_run_evaluation.py
```

Clustering needs at least two documents, and classifier training needs at least
two labeled classes. Edit `data/eval/queries.json` to add relevant document IDs
before interpreting search scores. The RAG questions in
`data/eval/questions.json` are editable examples.

## Optional Streamlit app

The repository also includes a Streamlit interface. Install Streamlit
separately, then launch it from the repository root:

```powershell
python -m pip install streamlit
streamlit run app/streamlit_app.py
```

The answer generator is extractive by default and uses sentences retrieved
from the corpus.

## Project layout

- `config.py`: paths and tunable pipeline defaults.
- `app/api.py`: FastAPI endpoints for search, documents, statistics, and similar decisions.
- `frontend/`: React/Vite Arabic RTL web interface.
- `app/streamlit_app.py`: optional Streamlit interface.
- `src/extraction/`, `src/morphology/`: source loading, metadata, Arabic text analysis.
- `src/indexing/`, `src/search/`, `src/similarity/`: lexical retrieval and metrics.
- `src/reduction/`, `src/clustering/`, `src/classification/`: course workflows.
- `src/rag/`, `src/evaluation/`: chunk retrieval, grounded answers, evaluation.
- `scripts/`: runnable pipeline entry points; `tests/`: focused regression tests.

Arabic stopword lists are plain UTF-8 files in `data/resources/`. The default
stemmer is a conservative light stemmer implemented in this project; stemming
can be disabled or switched to the optional NLTK ISRI stemmer. Metadata
extraction is heuristic and should be checked against the source collection.

Run the Python tests with:

```powershell
python -m pytest
```
