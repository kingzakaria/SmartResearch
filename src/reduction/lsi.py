"""Latent semantic indexing using truncated SVD."""

from collections.abc import Mapping, Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from config import DEFAULT_LSI_COMPONENTS, DEFAULT_RANDOM_STATE


def reduce_documents(
    documents: Mapping[str, Iterable[str]], components: int = DEFAULT_LSI_COMPONENTS
) -> tuple[list[str], TruncatedSVD, object]:
    if components < 1:
        raise ValueError("components must be at least 1")
    doc_ids = list(documents)
    texts = [" ".join(documents[doc_id]) for doc_id in doc_ids]
    if not texts or not any(texts):
        raise ValueError("LSI requires at least one non-empty document")
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(texts)
    n_components = min(components, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if n_components < 1:
        raise ValueError("LSI requires at least two documents and two distinct terms")
    model = TruncatedSVD(n_components=n_components, random_state=DEFAULT_RANDOM_STATE)
    return doc_ids, model, model.fit_transform(matrix)
