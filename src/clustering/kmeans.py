"""K-means clustering for tokenized documents."""

from collections.abc import Iterable, Mapping
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DEFAULT_RANDOM_STATE


def cluster_documents(
    documents: Mapping[str, Iterable[str]],
    n_clusters: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, int]:
    doc_ids = list(documents)
    if not 2 <= n_clusters <= len(doc_ids):
        raise ValueError("n_clusters must be between 2 and the number of documents")
    texts = [" ".join(documents[doc_id]) for doc_id in doc_ids]
    if not any(texts):
        raise ValueError("clustering requires non-empty analyzed documents")
    matrix = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(texts)
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(matrix)
    return dict(zip(doc_ids, map(int, labels)))
