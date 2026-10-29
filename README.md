# Arabic Legal Text Mining

A reproducible, offline-first text-mining project for Arabic legal decisions. It
includes document extraction, Arabic normalization and analysis, TF-IDF/BM25
search, clustering, supervised classification, a hybrid retrieval/RAG pipeline,
and evaluation utilities.

## Requirements and setup

Use Python 3.10 or later. From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put UTF-8 `.txt` decisions in `data/raw/real_estate_chamber/`. The project does
not include private case files. Keep source files unchanged; generated data and
models are written to `data/processed/` and `models/`.

## Run the pipeline

Run scripts from the repository root in order:

```powershell
python scripts/00_data_check.py
python scripts/01_extract.py
python scripts/02_analyze.py
python scripts/03_build_index.py
python scripts/04_search_cli.py
python scripts/05_cluster.py
python scripts/06_label.py
python scripts/07_train_classifier.py
python scripts/08_build_rag.py
python scripts/09_run_evaluation.py
```

The initial checkout has no decision files or labels. Extraction and analysis
report empty input without fabricating records; clustering needs at least two
documents, and classifier training needs at least two labeled classes. Edit
`data/eval/queries.json` to add relevant document IDs before interpreting search
scores. The RAG questions in `data/eval/questions.json` are editable examples.

Search can also be used interactively after building the index:

```powershell
python scripts/04_search_cli.py
```

## Streamlit app

Install the optional UI dependency and launch it from the repository root:

```powershell
python -m pip install streamlit
streamlit run app/streamlit_app.py
```

The app's search and retrieval features work offline. The answer generator is
extractive by default and only quotes sentences retrieved from the corpus; it
does not call an external LLM.

## Project layout

- `config.py`: paths and tunable pipeline defaults.
- `src/extraction/`, `src/morphology/`: source loading, metadata, Arabic text analysis.
- `src/indexing/`, `src/search/`, `src/similarity/`: lexical retrieval and metrics.
- `src/reduction/`, `src/clustering/`, `src/classification/`: course workflows.
- `src/rag/`, `src/evaluation/`: chunk retrieval, grounded answers, evaluation.
- `scripts/`: runnable pipeline entry points; `tests/`: focused regression tests.

Arabic stopword lists are plain UTF-8 files in `data/resources/`. The default
stemmer is a conservative light stemmer implemented in this project; stemming
can be disabled or switched to the optional NLTK ISRI stemmer. Metadata
extraction is heuristic and should be checked against the source collection.

Run tests with:

```powershell
python -m unittest discover -s tests -v
```
