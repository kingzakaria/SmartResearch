"""The three zones of the vocabulary (Zipf's law, Luhn's cut-offs).

Sort the terms by how many documents contain them (document frequency, df):

    frequent zone   df > MAX_DOC_FREQ_RATIO x |D|   present almost everywhere: they do not tell documents apart
    kept zone       in between                      the words that carry the topic of a document
    rare zone       df < MIN_DOC_FREQ               names, numbers, typos of the PDF conversion

The two ends are eliminated, the middle is kept. For SEARCH only the frequent end is cut (a rare word,
like a party name, is exactly what a user may look for); for CLUSTERING and CLASSIFICATION both ends are cut.
"""
FREQUENT, KEPT, RARE = "frequent", "kept", "rare"


def classify_terms(index, min_df: int = 2, max_ratio: float = 0.8) -> dict[str, str]:
    """term -> zone, from the document frequencies of an InvertedIndex."""
    limit = max_ratio * index.n_docs
    zones = {}
    for term in index.postings:
        df = index.df(term)
        zones[term] = FREQUENT if df > limit else RARE if df < min_df else KEPT
    return zones


def prune(terms_by_doc: dict, zones: dict[str, str], drop=(FREQUENT, RARE)) -> dict:
    """The same documents with the terms of the dropped zones removed."""
    drop = set(drop)
    return {d: [t for t in terms if zones.get(t, KEPT) not in drop] for d, terms in terms_by_doc.items()}


def summary(index, zones: dict[str, str]) -> dict:
    """Number of distinct terms and of occurrences (tokens) in each zone."""
    out = {z: {"terms": 0, "tokens": 0} for z in (FREQUENT, KEPT, RARE)}
    for term, zone in zones.items():
        out[zone]["terms"] += 1
        out[zone]["tokens"] += sum(index.postings[term].values())
    return out
