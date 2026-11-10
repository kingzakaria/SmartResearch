# Arabic court decisions: search, classification and web interface

A text-mining project (Introduction to Text Mining course) built on decisions of the Moroccan Court of Cassation written in Arabic, from two chambers: **real estate** (358 usable
decisions) and **personal status and inheritance** (second class, added later). It covers the whole chain taught in the course:
extraction, morphological analysis, inverted index and TF-IDF, similarity and distance measures, a search engine,
clustering, classification with cross-validation, a React interface, and a question-answering page that combines
retrieval and an LLM (RAG).

The formulas of the course (TF, IDF with log base 10, cosine, Jaccard, Dice, Euclidean, Manhattan, Levenshtein, chi-square)
are **written by hand** in the project, not taken from a library. scikit-learn is used only for the classifiers, and the
Gemini API only for the embeddings and the generated answers of "Ask AI".

> **Status.** Done and tested: extraction, morphology, index, search, partitioned search, duplicates, clustering,
> classification, API, interface and **"Ask AI" (RAG with Gemini)**. **Not done yet:** the quantitative evaluation of the
> search (precision@k, MAP) and of the RAG retrieval (recall@5) (see [Roadmap](#roadmap)).

---

## Contents

1. [What the application does](#what-the-application-does)
2. [Architecture](#architecture)
3. [Data](#data)
4. [Project structure](#project-structure)
5. [Installation (Windows)](#installation-windows)
6. [Running the pipeline](#running-the-pipeline)
7. [Running the application](#running-the-application)
8. [How each part works](#how-each-part-works)
9. [Results](#results)
10. [What we found and changed along the way](#what-we-found-and-changed-along-the-way)
11. [Tests](#tests)
12. [Troubleshooting](#troubleshooting)
13. [Limitations](#limitations)
14. [Roadmap](#roadmap)

---

## What the application does

| Capability | Description |
|---|---|
| **Search** | TF-IDF ranking with cosine similarity over an inverted index. A query is analysed with the same morphology as the documents, so `الشفعة`, `بالشفعة` and `شفعة` find the same decisions. Results show the matched words highlighted in the original text. |
| **Spelling help** | A word that is not in the corpus gets "did you mean" suggestions (Levenshtein distance). |
| **Similar decisions** | For any decision, the closest decisions with a choice of measure: cosine, Dice, Jaccard, Euclidean, Manhattan. |
| **Partitioned search** | The collection is divided into 3 groups with K-Means; a query can search only the best group(s): faster, with a measured loss of accuracy. |
| **Category filter** | Each decision has one of 5 categories (derived from its subject line); the search can be restricted to a category. |
| **Text classification** | Paste the reasoning of a decision and get the probability of each category. |
| **Ask AI** | A question in Arabic or English: hybrid retrieval (keywords + embeddings) finds passages of the decisions, Gemini writes the answer from those passages only and cites them as [1], [2]; every citation links to the decision. |
| **Chamber routing** | With two chambers (classes), a Naive Bayes classifier written by hand (the TP of the course) reads the query, predicts its chamber, and only that chamber is searched. If it is not sure, both are searched. The user can override it: Auto, All chambers, or one chamber. |
| **Web interface** | React single-page application in front of a FastAPI backend. Arabic text is rendered right-to-left in a Naskh typeface. |

## Architecture

```
                    data/raw/*.txt  (358 usable decisions, Arabic)
                              |
        [1] EXTRACTION   clean text, header metadata (decision no., date, case no., subject), body
                              |           documents.json
        [2] MORPHOLOGY   normalise -> tokenise -> stopwords -> stem        terms.json
                              |
        [3] INDEXING     inverted index {term -> {doc: count}} + TF-IDF     models/index.pkl
                              |
          +-------------------+---------------------------+
          |                                               |
 [4] SEARCH ENGINE                          [5] ANALYSIS OF THE COLLECTION
   query -> same analyse()                     duplicates / template families
   -> candidates -> cosine -> top k            vocabulary zones (frequent / kept / rare)
   highlight, suggestions, similar docs        K-Means -> 3 partitions (+ partitioned search)
          |                                    labels from subject lines -> classification
          |                                               |
          +-------------------+---------------------------+
                              |
                 [6] FastAPI (app/api.py)   ->   [7] React interface (frontend/)
                              |
        [8] CHAMBER ROUTING  query -> Naive Bayes -> predicted chamber -> search only its decisions (both if unsure)
                              |
        [9] ASK AI (RAG)  passages -> keyword + embedding retrieval -> fusion -> Gemini -> cited answer
```

Rule that holds everywhere: **documents and queries go through the same `analyze()`**. The keyword side uses stemmed
text; the text shown to the reader is never altered.

## Data

* **Two chambers (classes)**, one folder each, one `.txt` file per decision:
  * `data/raw/real_estate_chamber/`: 365 files, 7 empty, **358 usable** (الغرفة العقارية);
  * `data/raw/personal_status_chamber/`: the decisions of غرفة الأحوال الشخصية والميراث (the TP counts 1,883 files).
  The folders and their Arabic names are listed in `config.CHAMBERS`. Ids of the first chamber stay plain numbers (`12`),
  the others are prefixed (`ps-12`), so two files with the same number never collide.
* The data is **not in this repository** (real court decisions, kept out by `.gitignore`). Put the files in
  `data/raw/real_estate_chamber/` to rebuild everything. What the repository does contain in `data/` is
  `data/resources/` (stopwords and category rules): these are part of the code.
* PDF conversion left artefacts (scrambled lines, a Persian *yeh* in some files, headers in several formats). They are
  handled by the extraction step (metadata found in 99.7% of the files).

## Project structure

```
app/                     FastAPI backend (api.py)
config.py                every path and parameter of the project
.env                     your Gemini key (never committed; you create it)
data/
  raw/                   the decisions, one folder per chamber (not in the repository)
  processed/             generated: documents.json, terms.json, duplicates.json, labels.csv, chunks.json
  resources/             stopwords_ar.txt, stopwords_legal.txt, category_rules.json
frontend/                React (Vite) interface
models/                  generated: index.pkl, partitions.json, classifier.pkl, naive_bayes.json, chunk_embeddings.npz
reports/                 generated: CSV reports of the scripts
scripts/                 numbered scripts, run in order (see below)
src/
  extraction/            loader, cleaner (+ panel names removal), metadata parser
  morphology/            normalizer, tokenizer, stopwords, stemmer, isri (vendored), analyzer
  indexing/              inverted_index, tfidf, vocabulary (three zones)
  similarity/            measures (cosine, Jaccard, Dice, Euclidean, Manhattan), levenshtein
  search/                engine, highlight, partitions (partitioned search), chamber_router (Naive Bayes on the query)
  clustering/            kmeans (by hand), partitioning
  analysis/              duplicates, labels (chi-square, purity, ARI), categories (rules on subject lines)
  classification/        dataset (leak-free), evaluate (pipelines, cross-validation), service, naive_bayes (by hand)
  rag/                   chunker, gemini (HTTP client), retriever (hybrid + fusion), answer (prompt, citations), store
tests/                   179 automated tests
```

## Installation (Windows)

Requirements: Python 3.12 or newer (developed on 3.14), Node.js 18 or newer.

```powershell
cd C:\Users\hp\Documents\GitHub\SmartResearch
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

Put the decisions in `data\raw\real_estate_chamber\`, then run the pipeline below.

## Running the pipeline

Run from the project root with the virtual environment active. Each script reads the output of the previous ones.

| # | Command | What it does | Produces |
|---|---|---|---|
| 1 | `python scripts/01_extract.py` | Cleans the text, parses the header, cuts the closing formula | `data/processed/documents.json` |
| 2 | `python scripts/02_analyze.py` | Morphological analysis; compares the word / light / root modes | `terms.json`, `reports/mode_comparison.csv` |
| 3 | `python scripts/03_build_index.py` | Inverted index and TF-IDF | `models/index.pkl` |
| 4 | `python scripts/04_search_cli.py "الشفعة"` | Search from the command line (`-k`, `--all`, `--similar ID`, `--measure`) | console |
| 5 | `python scripts/05_find_duplicates.py` | Duplicates and template families | `duplicates.json`, `reports/duplicates.csv` |
| 6 | `python scripts/06_vocabulary_zones.py` | Frequent / kept / rare vocabulary | `reports/vocabulary_zones.csv` |
| 7 | `python scripts/07_build_partitions.py` | K-Means into 3 groups (`--explore` compares k = 2..8) | `models/partitions.json`, `reports/partitions.csv` |
| 8 | `python scripts/08_compare_partitioned.py` | Full search against partitioned search | `reports/partition_comparison.csv` |
| 9 | `python scripts/09_check_prefix_labels.py` | Checks whether the decision-number prefix is a category (it is not) | console |
| 10 | `python scripts/10_label_by_subject.py` | Category of each decision from its subject line | `data/processed/labels.csv` |
| 11 | `python scripts/11_train_classifiers.py` | Compares 5 classifiers with cross-validation, saves the best | `models/classifier.pkl`, `reports/classification_results.csv`, `reports/confusion_matrix.csv` |
| 12 | `python scripts/12_build_chunks.py` | Cuts the decisions into passages of about 400 words (overlap 80), skips duplicates | `data/processed/chunks.json` |
| 13 | `python scripts/13_embed_chunks.py` | One Gemini embedding per passage; resumable (needs the key) | `models/chunk_embeddings.npz` |
| 14 | `python scripts/14_ask_cli.py "سؤال"` | Ask from the terminal (`--no-llm`: show only the retrieved passages) | console |
| 15 | `python scripts/15_train_naive_bayes.py` | Naive Bayes chamber classifier: TP-style test, query-level cross-validation, final model | `models/naive_bayes.json` |
| 16 | `python scripts/16_compare_routed_search.py` | Search in every chamber against search routed by Naive Bayes | `reports/routed_search_comparison.csv` |

Minimal order to get the full application: **1, 2, 3, 5, 7, 10, 11, 15** (+ 12 and 13 for "Ask AI").
Adding the second chamber to an existing project: put its files in `data/raw/personal_status_chamber/`, then run
**1, 2, 3, 5, 15** again (7 and 10 only work on the real-estate chamber, whatever the number of chambers; 12 and 13 are
not needed again). Script 01 prints the number of files found per chamber: check it.
Changing the stopwords, the cleaning or the category rules means re-running the scripts from that point on.

Helper scripts: `demo_morphology.py` (shows each step of the analysis on a sentence), `inspect_formulas.py`
(reproduces the worked examples of the course), `02_inspect_failures.py` (headers the parser could not read).

## Running the application

Two terminals, both from the project folder.

```powershell
# terminal 1: the API
.venv\Scripts\activate
python -m uvicorn app.api:app --reload --port 8000
```

```powershell
# terminal 2: the interface
cd frontend
npm run dev
```

Open <http://localhost:5173>. The interactive API documentation is at <http://localhost:8000/docs>.

The API degrades gracefully: without `partitions.json` the group search is hidden, without `labels.csv` there is no
category filter, without `classifier.pkl` the classify page explains which script to run.

### Gemini key for "Ask AI"

1. Create a free key in [Google AI Studio](https://aistudio.google.com/apikey).
2. Create the file `.env` at the project root (next to `config.py`) containing one line: `GEMINI_API_KEY=your_key_here`
   (PowerShell: `Set-Content -Path .env -Value "GEMINI_API_KEY=your_key_here" -Encoding ascii`).
3. Make sure `.env` is in `.gitignore` (it is). **Never commit the key.**
4. Run scripts 12 and 13, then start the API.

Settings in `config.py`:

| Setting | Default | Role |
|---|---|---|
| `GEMINI_LLM_MODEL` | `gemini-flash-latest` | model that writes the answer (an alias that follows Google's current Flash model) |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | model that embeds passages and questions |
| `EMBED_DIM` | 768 | size of the embedding vectors |
| `GEMINI_TIMEOUT` | 180 | seconds to wait for one answer |
| `RAG_TOP_PASSAGES` / `RAG_MAX_PER_DOC` | 5 / 2 | passages given to the LLM / at most this many from one decision |

Model names change. To list what your key can use:

```powershell
$key = (Get-Content .env | Select-String "GEMINI_API_KEY=").ToString().Split("=")[1]
(Invoke-RestMethod "https://generativelanguage.googleapis.com/v1beta/models?key=$key&pageSize=100").models | Where-Object { $_.supportedGenerationMethods -contains "generateContent" } | Select-Object -ExpandProperty name
```

Free-tier limits apply (see [Troubleshooting](#troubleshooting)): script 13 pauses between requests, retries on network
errors, "429" and "503", and saves its progress after every batch, so it can simply be run again.

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health`, `GET /api/stats` | Size of the collection, mode, availability of partitions and classifier |
| `GET /api/search?q=&k=&all=&n_probe=&category=&chamber=` | Search. `all=true` requires every word; `n_probe=1` searches only the best group (real-estate chamber); `category=` filters; `chamber=` is `auto` (Naive Bayes predicts it), `all` or a chamber key; the answer says which chambers were searched and why |
| `GET /api/chambers` | The chambers with their Arabic name and number of decisions |
| `GET /api/documents/{id}?q=` | Metadata, subject, text with the matched words marked, closing formula, category |
| `GET /api/documents/{id}/similar?k=&measure=&same_chamber=` | Similar decisions (by default inside the decision's own chamber) |
| `GET /api/partitions` | The groups: size and main terms |
| `GET /api/categories` | Categories with their Arabic name and number of decisions |
| `POST /api/ask` | `{ "question": "..." }` returns the answer, the retrieval mode and the passages used (with the decision they come from and whether the answer cites them) |
| `POST /api/classify` | `{ "text": "..." }` returns a probability for each category |

## How each part works

### 1. Extraction and cleaning (`src/extraction`)

* `clean_text`: unifies line breaks, removes invisible characters, converts Arabic-Indic and Persian digits and the Persian
  *yeh*, drops letterhead lines (`المملكة المغربية`, `محكمة النقض`...).
* `parse_document`: finds the decision number, date, case number and **subject line** in the header, in all the formats
  met in the corpus (including typos such as `رتم` for `رقم`).
* `split_closing`: cuts the closing formula (`وبه صدر القرار وتلي...`, with the names of the judges and the clerk) from the
  body.
* `strip_panel_names`: removes the names of the rapporteur, the advocate general, the clerk and the panel from the text used
  for analysis (the reader still sees the original text).

### 2. Morphology (`src/morphology`)

`normalize -> tokenize -> stopwords -> stem -> drop boilerplate stems`

* **Normalisation**: diacritics and tatweel removed; alef, yeh, teh marbuta forms unified.
* **Tokenisation**: Arabic letters or digits; numbers kept only if they have 2 or 3 digits.
* **Stopwords** from two external files: generic Arabic words and the legal boilerplate of the decisions
  (المحكمة, الطاعن, ...), plus the month names. A word is removed when its surface form **or its stem** is listed.
* **Stemming**, three modes compared by `02_analyze.py`:
  * `word`: no stemming;
  * `light` (default): hand-written rules, removes the article and clitic prefixes (`وال بال كال فال لل ال`, `و ب ل` when
    safe) and the suffixes (`ها هم ان ات ون ين يه ه ي`), with minimum lengths and a list of protected words
    (`قانون`, `عنوان`...);
  * `root`: ISRI root extraction (vendored copy of the algorithm of NLTK's `ISRIStemmer`, Apache-2.0, so that the project does
    not depend on NLTK, which does not run on Python 3.14).

### 3. Inverted index and TF-IDF (`src/indexing`)

```
TF(i,j)  = w(i,j) / |d_j|             IDF(i) = log10( |D| / df(i) )          weight = TF x IDF
```

* The index stores `term -> {document: occurrences}`, the length of each document and the document frequency.
* Vectors are sparse dictionaries. Log base 10 is required to reproduce the worked example of the course
  (there is a test for it).
* A term present in every decision has IDF 0 and therefore no weight (8 such terms).

### 4. Similarity and distances (`src/similarity`)

| Measure | Formula | Type |
|---|---|---|
| Cosine | `q.d / (‖q‖ ‖d‖)` | similarity on vectors (used for ranking) |
| Jaccard | `‖A ∩ B‖ / ‖A ∪ B‖` | similarity on sets |
| Dice | `2‖A ∩ B‖ / (‖A‖ + ‖B‖)` | similarity on sets |
| Euclidean | `sqrt(sum (q_i - d_i)^2)` | distance |
| Manhattan | `sum ‖q_i - d_i‖` | distance |
| Levenshtein | minimum insertions, deletions, substitutions | distance on words (spelling suggestions) |

### 5. Search engine (`src/search`)

`query -> analyze() -> terms in the index? -> candidate documents (those sharing a term) -> TF-IDF vector of the query ->
cosine -> sort -> top k`. Options: AND mode, "did you mean", the list of words ignored because they appear in every
decision, highlighted snippets built on the **original** words of the text.

### 6. Duplicates and template families (`src/analysis/duplicates.py`)

Two decisions are compared by the Jaccard similarity of their term sets.

* **Duplicate** = the same decision published twice: same case number and similarity of at least 0.8 (or a nearly identical
  text, 0.97). One copy is set aside: **31 duplicates, 327 decisions remain**.
* **Template family** = different decisions that follow the same template (for example the decisions 2/67 to 2/71, with
  consecutive case numbers). They are kept, but cross-validation never separates a family: **3 families**, the largest with
  9 decisions.

### 7. The three zones of the vocabulary (`src/indexing/vocabulary.py`)

Zipf's law and Luhn's cut-offs: the terms are sorted by the number of decisions containing them.

| Zone | Rule | Content |
|---|---|---|
| frequent | in more than 80% of the decisions (50% for clustering and classification) | boilerplate that cannot tell decisions apart |
| kept | in between | the words that carry the topic |
| rare | in fewer than 2 decisions | names, numbers, conversion errors |

On the corpus the rare zone holds 41% of the distinct terms but only 2.4% of the occurrences. For **search** only the
frequent end is cut (a rare word, such as a party name, is exactly what a user may look for); for **clustering and
classification** both ends are cut.

### 8. K-Means and partitioned search (`src/clustering`, `src/search/partitions.py`)

* K-Means written by hand (spherical: the similarity is the cosine), k-means++ initialisation, 10 restarts, empty clusters
  restarted. Features: TF-IDF on the cleaned vocabulary of the 327 non-duplicate decisions. The 31 duplicates join the group
  of their twin.
* **Partitioned search**: the query is compared with the 3 centroids; only the `n_probe` closest groups are searched, each
  with its own inverted index. Documents are scored with the same TF-IDF model as the full search, so a decision has the
  same score in both modes; only the set of documents looked at changes. If no query word belongs to the clustering
  vocabulary, every group is searched.
* The groups correspond to topics: land registration, eviction and possession, and pre-emption, partition, sale and
  reopening of judgments.

### 9. Categories and classification (`src/analysis/categories.py`, `src/classification`)

**Labels.** Each decision gets a category from its **subject line** with keyword rules (`data/resources/category_rules.json`,
first match wins, the part of the subject before the first dash is tried first, matching is done on stems). Five
categories:

| Category | Arabic name | Decisions |
|---|---|---|
| registration | التحفيظ العقاري | 117 |
| procedure | المسطرة والطعون وتنظيم المحاكم | 75 |
| property_contracts | الملكية والعقود والاستحقاق | 56 |
| eviction | الإفراغ والكراء والحيازة | 34 |
| preemption_partition | الشفعة والقسمة والشياع | 33 |

Twelve decisions without a usable subject stay "other" and are not used for training.

**Why the subject and not the clusters.** The subject is written by a human and names the topic, and the classifier only sees
the reasoning, so it cannot read its own label. Labelling by the K-Means groups would be circular (the same text features
would produce both the label and the prediction).

**No leakage.** The text given to the classifier is the body **without the subject lines** that open it, without the panel
names and without bare numbers. `scripts/11_train_classifiers.py` measures the leak (subject still present in the text):
**1.0%**.

**Evaluation.** 5-fold cross-validation, repeated 3 times with different splits, **stratified** (each fold has every category)
and **grouped** (a duplicate or a template family is never split between training and test). Frequency thresholding and the
chi-square selection are fitted inside each training fold. Models: Complement Naive Bayes, Rocchio (nearest centroid), KNN
(cosine), logistic regression, linear SVM; each keeping 10%, 25% or 100% of the terms ranked by chi-square.

### 10. Chamber routing with Naive Bayes (`src/classification/naive_bayes.py`, `src/search/chamber_router.py`)

When the collection holds several classes, searching all of it is slower and mixes vocabularies. The classifier of the TP
reads the query first:

```
P(c)    = documents of class c / all documents
P(w|c)  = ( n(w,c) + 1 ) / ( N_c + m )                 Laplace smoothing; m = distinct words of the training corpus
c_MAP   = argmax  log P(c) + sum over the words of the query of log P(w|c)        sum of logarithms: no underflow
```

* Written by hand like the other formulas of the course. The text is analysed by the same `analyze()` as the documents
  (stems), duplicates are left out, and words unknown to the model are ignored. The model stores the counts
  `n(w,c)`, and `P(w|c)` is computed on demand (same numbers as the TP, a much smaller file). A test reproduces the numbers of
  the movie-review example of the course.
* **Routing rule**: the best chamber is searched alone only if its probability is at least `CHAMBER_MIN_CONFIDENCE` (0.90, chosen from the threshold table of script 15);
  with no known word or a doubtful query, every chamber is searched, so the loss of recall stays limited to confident mistakes.
* **Same scores**: the TF-IDF model is global, so a decision has the same score with or without routing; routing only decides
  which decisions are scored (`TfidfModel.cosine_scores(..., allowed=...)`).
* **Priors**: in the TP they come from the number of decisions (here about 86% / 14%). `NB_PRIORS` in `config.py` is `"empirical"`
  (the TP, the default) or `"uniform"` (both chambers equally likely before reading the query). Uniform priors looked more
  fair for small chambers, but on the real data they were worse on every measure (see Results), so the TP's choice is kept.
  The probabilities are over-confident on long texts (the independence assumption is false): read them as "which class wins".
* **Evaluation** (`scripts/15_train_naive_bayes.py`): (1) the TP split, last 20 decisions of each chamber as test, confusion matrix;
  (2) what the search needs, queries built from held-out decisions (their subject, and 3 words of it), 5-fold cross-validation:
  accuracy, how often the model is sure, accuracy when it is sure, and the number of confident mistakes.
  `scripts/16_compare_routed_search.py` then measures routed against full search: share narrowed, top 10 kept, decisions scored, time.
* Interface: a Chamber selector (Auto, All chambers, one chamber). In Auto the page says which chamber was searched, how sure the
  classifier was, and offers "Search in all chambers instead".
* What is limited to the real-estate chamber: the 5 subject categories (rules written for its subjects), the K-Means groups
  and "Ask AI". The category filter turns the automatic routing off.

### 11. Ask AI: RAG (`src/rag`)

```
question -> [keyword retriever: TF-IDF cosine over passages] --\
         -> [embedding retriever: Gemini vectors, cosine]    ----> reciprocal rank fusion -> top 5 passages
         -> prompt (rules + numbered passages + question) -> Gemini -> answer with [n] citations -> sources
```

* **Passages** (`chunker.py`): about 400 words, overlap of 80, cut at the end of a sentence when possible; the subject line
  and the duplicates are left out.
* **Keyword retriever**: the same engine as the search page (same morphology, same hand-written TF-IDF and cosine) applied to the passages.
* **Embedding retriever**: `gemini-embedding-001` through the HTTP API (768 dimensions, normalised so that a dot product is a cosine).
  Passages and questions use different task types (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`).
* **Fusion**: reciprocal rank fusion, `score = sum 1/(60 + rank)`: only ranks are used, so the two score scales are never compared.
  At most 2 passages per decision keep the sources varied.
* **Generation** (`answer.py`): the LLM must answer only from the numbered passages, cite them, and say when they do not
  contain the answer. With no retrieved passage the LLM is not called.
* **Fallback**: without embeddings, or if Gemini's embedding call fails, the answer uses keywords only (the interface says so).
* **Why the Gemini API for embeddings and not a local model**: the local multilingual models need PyTorch, which is heavy
  and not always available for the newest Python on Windows; the API needs one `httpx` call and the same key as the LLM.
* **Interface**: citations such as [1] in the answer are links to the numbered passage below it; each passage links to the
  full decision and says whether the answer cited it. Each question is independent (no memory of previous turns).
* **Reliability of the client** (`gemini.py`): plain HTTPS with `httpx`; retries with growing pauses on dropped connections,
  timeouts, "429" and "503"; a missing key, a missing `chunks.json` or an unknown model give a message that says what to do.

### 12. Web interface (`frontend/`)

React 18 + Vite, plain CSS, react-router. All interface text is in `src/strings.js`. The search state is kept in the URL
(`?q=...&category=...&scope=...`), so the back button and shared links work. The header and the text of the decisions use
the Amiri and Noto Naskh Arabic typefaces; the interface uses IBM Plex Sans Arabic.

## Results

### Collection

| | Real estate | Personal status | Total |
|---|---|---|---|
| Files | 365 | 1,883 | 2,248 |
| Usable decisions (7 empty files skipped) | 358 | 1,883 | 2,241 |
| After removing duplicates | 327 | 1,878 | 2,205 |

| | |
|---|---|
| Vocabulary (stems, light mode) | 24,132 terms (word mode: 49,793; root mode: 9,783) |
| Average document length | 521 terms |
| Postings (term, document pairs) | 664,275 |

The light stemmer reduces the vocabulary by 51.5% compared with no stemming; the root mode by 80.4%.
Duplicates: 36 groups (same case number and nearly identical text); template families are kept and never split by cross-validation.

Metadata coverage is uneven: the headers of the second chamber are written differently, so only 22.4% of all decisions
have a decision number (date 87.4%, case number 92.3%, subject 89.1%). This does not affect the search or the routing, which use the text.

### Chamber routing with Naive Bayes

Training set: 1,878 personal-status and 327 real-estate decisions (duplicates set aside), priors 85.8% / 14.2%.

**TP-style test**: the last 20 decisions of each chamber as test, the rest (2,165) as training, vocabulary m = 23,818:
accuracy **100% (40 of 40)**. It is a small and easy test: the two chambers are clearly different on whole decisions.

**Held-out queries** (what the search needs): the subject of each decision, and 3 words of it, 5-fold cross-validation
(1,955 queries of each kind), confidence threshold 0.80:

| Query | Priors | Accuracy | Sure | Right when sure | Sent to the wrong chamber |
|---|---|---|---|---|---|
| subject | empirical | 89.0% | 84.9% | 94.6% | 90 |
| subject | uniform | 84.5% | 80.7% | 91.2% | 139 |
| 3 words | empirical | 89.1% | 80.8% | 94.9% | 81 |
| 3 words | uniform | 80.3% | 66.3% | 92.7% | 94 |

The priors of the TP (the share of decisions of each chamber) are better than equal priors on every column, so they are the default.
Raising the confidence threshold trades narrowed queries against queries sent to the wrong chamber (both kinds of query, empirical priors):

| Threshold | Queries narrowed to one chamber | Right when narrowed | Wrong chamber, % of all queries |
|---|---|---|---|
| 0.60 | 95.4% | 90.8% | 8.8% |
| 0.80 | 82.8% | 94.7% | 4.4% |
| **0.90 (chosen)** | **72.4%** | **97.3%** | **2.0%** |
| 0.95 | 65.1% | 98.0% | 1.3% |
| 0.99 | 51.6% | 99.2% | 0.4% |

**Routed search against search in both chambers** (`scripts/16_compare_routed_search.py`, 3,910 queries, threshold 0.90):

| Queries | Narrowed | Right chamber | Top 10 kept | Top 1 kept | Decisions scored | Time (both -> routed) |
|---|---|---|---|---|---|---|
| all | 72.4% | 97.9% | 96.7% | 97.1% | 85.6% | 11.97 -> 11.03 ms |
| subject | 77.8% | 97.7% | 96.2% | 97.0% | 82.4% | 13.92 -> 12.53 ms |
| 3 words | 66.9% | 98.2% | 97.2% | 97.2% | 88.8% | 10.01 -> 9.53 ms |
| narrowed ones only | 100% | 97.9% | 95.4% | 96.0% | 80.0% | 12.05 -> 10.83 ms |
| searched in both (not sure) | 0% | | 100% | 100% | 100% | |

Reading: routing keeps 96.7% of the top 10 of the full search, scores 14% fewer decisions and is about 8% faster. The gain in time is small
at this size; the main benefit is that the results come from the chamber the query is about, with "All chambers" one click away.
This comparison is slightly optimistic, because the saved model was trained on the decisions the queries come from; the held-out
tables above are the honest accuracy. The two chambers overlap in topics (gifts, partition, registered property), so part of the
mistakes are genuinely ambiguous queries.

### Partitioned search against full search

Measured by `08_compare_partitioned.py` inside the real-estate chamber (358 decisions, 3 groups), 313 queries (the subject line of each decision), top 10:

| Groups searched | Top 10 kept | Top 1 kept | Documents scored | Time (full -> partitioned) |
|---|---|---|---|---|
| 1 of 3 | 73.5% | 84.0% | 36.3% | 0.97 ms -> 0.31 ms |
| 2 of 3 | 92.1% | 96.5% | 67.0% | 0.82 ms -> 0.48 ms |
| 3 of 3 | 100% | 100% | 100% | identical to the full search |

The last row is a correctness check: searching every group reproduces the full search exactly. Absolute times are below a
millisecond because the collection is small; the saving is expected to matter on a much larger collection.
The groups cover the real-estate chamber only, so the script compares full and partitioned search **inside that chamber**.

K-Means quality is low, as usual on legal text where every decision shares the same wording: silhouette between 0.03 and
0.05 for k = 2..8. The groups are readable topics, but their borders move when the vocabulary changes slightly
(see [Limitations](#limitations)).

### Ask AI (RAG)

| | |
|---|---|
| Decisions used | 327 (the 31 duplicates are skipped, so the same text is never retrieved twice) |
| Passages | 1,047 (1 to 12 per decision, median 3) |
| Words per passage | median 380, maximum 400 |
| Embeddings | 1,047 vectors of 768 dimensions |

First test, the question "ما هي شروط الشفعة؟" (what are the conditions of pre-emption): retrieval was hybrid, the answer was
written in Arabic with citations [1] to [5] that pointed to decisions 4/26, 2/50 and 2/52, and it covered the deposit of the
price, the offer of the price and costs, the legal deadline, several pre-emptors and the party the action is brought against.
This is a single check, not an evaluation; the measured comparison of the retrievers is in the [Roadmap](#roadmap).

### Classification

315 decisions, 5 categories. Baseline (always the biggest category): **accuracy 37.1%, macro-F1 0.108**.

Best model: **logistic regression, accuracy 78.8%, macro-F1 0.766** (about twice the baseline in accuracy). The other four
models (Naive Bayes, Rocchio, KNN, linear SVM) are in `reports/classification_results.csv`; the spread across folds is
about ±0.06, so the best models are statistically close.

Per category, the weakest is **property_contracts (F1 0.66)**, and **procedure** has the lowest recall (0.67). Full
tables: `reports/classification_results.csv` and `reports/confusion_matrix.csv`.

Observations:

* The chi-square selection and the upper frequency limit did **not** improve accuracy here. On a small collection they reduce
  size and time, not error.
* The most weighted words of each category make legal sense: eviction `افراغ, احتلال, سكن`; pre-emption/partition
  `قسم, شياع, شفع, شركاء`; registration `متعرض, مطلب, كناش, محافظ, تقييد`; property/contracts `استحقاق, هبه, صدق, تعويض`.
* The labels come from keyword rules on the subject line: the scores measure how well the text predicts **that labelling**,
  not an absolute truth.

## What we found and changed along the way

The project did not go straight; these are the problems met and what was decided, since they explain the final design.

| Finding | Decision |
|---|---|
| Headers came in many formats (case number without `رقم`, typos, subject before the title line, subject merged with the reasoning) | The metadata parser was rewritten: 99.7% coverage |
| The closing formula and letterhead lines (with the names of the judges) polluted the text | Letterheads are dropped and the closing formula is cut from the body (355 of 358 found) |
| NLTK 3.4.5 does not run on Python 3.14 | The ISRI root stemmer is vendored |
| The boilerplate words (المحكمة, الطاعن...) were everywhere | External legal stopword list applied by surface and by stem; IDF 0 neutralises the rest. Words that are also topics (`بناء`, `القانون`) were deliberately not listed |
| First duplicate rule (similarity >= 0.8) grouped decisions 2/67 to 2/71, which are different cases with the same template | Split into **duplicates** (same case number, dropped) and **template families** (kept, never split by cross-validation) |
| "Is the first number of the decision number (1/2/3/4) a category?" | `09_check_prefix_labels.py`: the words linked to each prefix were judges' names, months and drafting phrases, not topics. **The prefix identifies the panel and session, not the subject**, so it is not used as a label |
| Judges' names leaked into the vocabulary (and would have leaked into clustering) | Panel names are cut out of the analysed text by anchored patterns (`strip_panel_names`); month names added to the stopwords |
| Clusters were first considered as labels | Rejected as circular; labels come from the subject lines instead |
| A decision's body begins with its own subject line | The subject is stripped from the classification text, and a leak check measures what remains (1.0%) |
| Some categories were too small to evaluate (below 20 decisions) | Pre-emption merged with partition; contracts merged with property |
| First API version loaded the classifier lazily without a lock; two simultaneous first requests could get a 503 | A lock was added, with a test that fires concurrent requests |
| A test assumed no trained model existed on the machine | The test now points explicitly to a missing file |
| Local multilingual embedding models need PyTorch, heavy and not always available for the newest Python on Windows | Embeddings and answers come from the Gemini API through plain `httpx`; with no embeddings the system still works with keywords only |
| The Gemini free tier allows 1,000 embedding requests per day and the connection dropped during the first long run | The embedding script is resumable, saves after every batch, uses small batches with pauses, and the client retries dropped connections and "429" / "503" |
| `gemini-2.5-flash` was refused with "no longer available to new users", and another model answered "503 high demand" | The model is a setting; the default is the alias `gemini-flash-latest` and the README shows how to list the models available to a key |
| Second chamber added: should the query decide which chamber to search? | Hand-written Naive Bayes (the TP) on the stems of the query; a single chamber is searched only if the best probability reaches 0.90, otherwise both |
| Equal priors looked fairer to the small chamber | Measured on held-out queries: worse on every column, so the priors of the TP (share of decisions) are kept |
| Confidence threshold 0.80 sent 4.4% of all queries to the wrong chamber | Threshold 0.90: 2.0% wrong, still 72% of queries narrowed |
| With two chambers the K-Means comparison would have compared the full search with itself (the groups exist for one chamber only) | Script 08 now stays inside the chamber the groups were built for; the engine uses the groups only for a search inside that chamber |
| A hard-to-spot mistake in `.gitignore`: `data/` ignored `data/resources/` too (stopwords, category rules), so a fresh clone could not run | The rule now ignores only `data/raw/` and the generated folders |

## Tests

```powershell
pytest
```

179 tests, no data needed (they use small in-memory corpora): extraction and cleaning, header parsing, morphology, index and
the numbers of the course, similarity measures, search engine, partitions and K-Means, duplicates and vocabulary zones,
labels and chi-square, categories, classification (including the leak guards and the fold grouping), the RAG (chunking,
fusion, hybrid retrieval, prompt and citations, the Gemini client with a fake HTTP layer, no network needed) the Naive Bayes (it reproduces the numbers of the course example), the routing over two chambers and the API.
The API and classification tests are skipped automatically if FastAPI or scikit-learn is not installed.

## Troubleshooting

| Message or symptom | Cause and fix |
|---|---|
| `503 ... No Gemini API key` (API) | `.env` is missing or misplaced: it must be next to `config.py`; restart the API after creating it |
| Script 15: `Only one chamber found` | the files of the second chamber are not in `data/raw/personal_status_chamber/` (or script 01 was not run again) |
| Search results never come from the second chamber | `models/naive_bayes.json` is missing or old: run script 15 and restart the API; or choose "All chambers" |
| `503 ... No passages yet` | run `python scripts/12_build_chunks.py`, then `13_embed_chunks.py` |
| `model ... is no longer available` / `not found` | change `GEMINI_LLM_MODEL` in `config.py` (see the list command above) |
| `503 ... high demand` | Google's side, temporary: ask again, or use another model (for example `gemini-3.1-flash-lite`) |
| `429` or "exceeded your current quota ... per day" while embedding | free-tier limit (1,000 embedding requests per day, a few hundred tokens per minute): wait, then run script 13 again, it continues |
| `Cannot reach Gemini ... timed out` / `getaddrinfo failed` | network problem: run the script again; the timeout can be raised with `GEMINI_TIMEOUT` |
| The answer says retrieval is "keywords only" | the embeddings file is missing, does not match `chunks.json` (run 12 then 13 again), or the embedding call failed; the interface shows the reason |
| The search page shows no category filter or no group choice | `labels.csv` or `partitions.json` does not exist yet (scripts 10 and 7) |

## Limitations

* **Small collection** (358 decisions): timings are tiny and the classification scores have a spread of about
  ±0.06.
* **Labels are rule-based.** Some subjects are matched by a generic keyword; the "procedure" and "property_contracts"
  categories are the least coherent. About 4% of the decisions have no category.
* **The K-Means groups are not stable**: their sizes changed between runs when the vocabulary changed slightly (silhouette
  0.03 to 0.05). They are interpretable topics, not strict classes.
* **Stemming is rule-based**: broken plurals (`الشفعاء`) get a different stem from the singular (`شفعة`) and need to be
  listed explicitly where it matters (the category rules do this).
* Searching for the **name of a judge** finds nothing, by design: those names are removed from the analysed text.
* **"Ask AI" can be wrong**: the LLM may misread a passage or cite one that does not support the claim; the citations exist so the
  reader can check. It is not legal advice. The passages and the questions are sent to Google's API, so use it only with data you may share. The free tier has daily and per-minute limits.
* **Routing mistakes**: about 2% of the queries are sent to the wrong chamber with confidence (held-out estimate at threshold 0.90), which hides their decisions until the user chooses "All chambers". The two chambers share topics (gifts, partition, registered property), and short generic queries are the hardest.
* **Second-chamber metadata**: only 22.4% of the decisions have a parsed decision number (the headers of the second chamber differ); the search and the routing do not depend on it, but the interface shows no number for them.
* The categories, the groups and "Ask AI" cover the real-estate chamber only: the second chamber has no subject rules, and embedding its passages would take several days of the free quota.
* The classifier always picks one of its five categories: it has no "none of these" answer, and a text from another field
  gets low probabilities rather than a refusal.

## Roadmap

0. **Second chamber everywhere**: category rules for its subjects, its own K-Means groups, and its passages in "Ask AI".
1. **Search evaluation**: 10 to 15 queries with their relevant decisions, precision@k and MAP, and a comparison of the
   word / light / root modes.
2. **Ranking improvements**: a boost for the subject line, or BM25; the decisions about "التشطيب" are ambiguous (striking a
   case off the roll against striking a registration) and rank poorly.
3. **RAG evaluation** (the next step to give the retrieval a number): 10 to 15 questions with the decisions that should be retrieved (recall@5 for keywords only, embeddings only
   and hybrid), and a check that every claim of an answer is supported by its cited passage. Memory of previous turns.
4. **Stability test of the clustering**: run K-Means with several seeds and measure the agreement between the runs; name the
   groups in the interface.

## Credits

* Course: *Introduction au Text Mining*.
* Gemini API (Google) for embeddings and answers.
* ISRI stemmer: algorithm of the NLTK `ISRIStemmer` (Apache License 2.0), vendored in `src/morphology/isri.py`.
* Typefaces (through `@fontsource`): Amiri, Noto Naskh Arabic, IBM Plex Sans Arabic.