"""K-Means, partitions and partitioned search. Run: pytest tests/test_partitions.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.clustering.kmeans import kmeans, silhouette, similarity_matrix  # noqa: E402
from src.clustering.partitioning import build_partitions  # noqa: E402
from src.indexing.inverted_index import InvertedIndex  # noqa: E402
from src.indexing.tfidf import TfidfModel  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402
from src.search.engine import SearchEngine  # noqa: E402

# two clear topics: sale / pre-emption (docs 1-6) and land registration (docs 7-12); doc 13 is a copy of doc 1
TERMS = {str(i): ["شفع", "بيع", "عقار", f"x{i % 3}"] for i in range(1, 7)}
TERMS.update({str(i): ["تحفيظ", "تعرض", "مطلب", f"y{i % 3}"] for i in range(7, 13)})
TERMS["13"] = list(TERMS["1"])
DOCS = [{"doc_id": d, "filename": f"{d}.txt", "decision_no": d, "date": None, "case_no": None,
         "subject": f"s{d}", "body": " ".join(t)} for d, t in TERMS.items()]


def vectors():
    model = TfidfModel(InvertedIndex.build({d: t for d, t in TERMS.items() if d != "13"}))
    return {d: model.doc_vector(d) for d in model.index.doc_len}


def test_kmeans_separates_two_topics_and_is_deterministic():
    assign, centroids, cohesion = kmeans(vectors(), 2, seed=1)
    assert len({assign[str(i)] for i in range(1, 7)}) == 1 and len({assign[str(i)] for i in range(7, 13)}) == 1
    assert assign["1"] != assign["7"] and cohesion > 0.5 and len(centroids) == 2
    assert kmeans(vectors(), 2, seed=1)[0] == assign


def test_kmeans_rejects_a_bad_k():
    with pytest.raises(ValueError):
        kmeans(vectors(), 0)


def test_silhouette_is_high_for_the_right_k_and_lower_for_a_bad_one():
    ids, sims = similarity_matrix(vectors())
    good = silhouette(ids, sims, kmeans(vectors(), 2)[0])
    bad = silhouette(ids, sims, {d: int(d) % 2 for d in ids})
    assert good > 0.4 and bad < 0 < good


def make_partitions(k=2):
    return build_partitions(TERMS, k, drop=["13"], duplicate_groups=[["1", "13"]], min_df=1, max_ratio=1.0)


def test_build_partitions_puts_the_dropped_copy_with_its_twin():
    data = make_partitions()
    a = data["assignments"]
    assert a["13"] == a["1"] and a["1"] != a["7"] and sum(data["sizes"]) == 13


@pytest.fixture(scope="module")
def engine():
    analyzer = analyzer_from_config(config, "light")
    terms = {d["doc_id"]: analyzer.analyze(d["body"]) for d in DOCS}
    index = InvertedIndex.build(terms)
    eng = SearchEngine(analyzer, TfidfModel(index), DOCS)
    eng.load_partitions(make_partitions())
    return eng


def ids(response):
    return [r.doc_id for r in response.results]


def test_best_group_gives_the_same_scores_as_the_full_search(engine):
    full, part = engine.search("شفعة بيع", k=20), engine.search("شفعة بيع", k=20, n_probe=1)
    assert part.routing["fallback"] is False and part.routing["groups"] == [engine.partitions.group_of("1")]
    assert set(ids(part)) == set(ids(full)) and ids(part)[:3] == ids(full)[:3]
    assert part.scored <= full.scored


def test_searching_the_other_topic_routes_to_the_other_group(engine):
    part = engine.search("مطلب التحفيظ", n_probe=1)
    assert part.routing["groups"] == [engine.partitions.group_of("7")] and set(ids(part)) == {str(i) for i in range(7, 13)}


def test_all_groups_equal_full_search(engine):
    full = engine.search("شفعة تحفيظ", k=20)
    every = engine.search("شفعة تحفيظ", k=20, n_probe=2)
    assert ids(every) == ids(full) and every.scored == full.scored


def test_a_group_search_can_miss_documents_of_the_other_group(engine):
    full, part = engine.search("شفعة تحفيظ", k=20), engine.search("شفعة تحفيظ", k=20, n_probe=1)
    assert len(part.results) < len(full.results)


def test_query_without_known_clustering_term_searches_everything(engine):
    groups, _, fallback = engine.partitions.route(["غريب"], n_probe=1)   # not in the clustering vocabulary
    assert fallback is True and groups == [0, 1]


def test_results_carry_their_group_and_default_is_unchanged(engine):
    assert engine.search("شفعة").routing is None
    assert {r.group for r in engine.search("شفعة").results} == {engine.partitions.group_of("1")}


def test_partitions_for_other_documents_are_ignored(engine):
    other = SearchEngine(engine.analyzer, engine.model, DOCS[:5])
    data = make_partitions()
    with pytest.warns(UserWarning):
        other.load_partitions(data | {"assignments": {"1": 0, "zzz": 1}})
    assert other.partitions is None and other.search("شفعة", n_probe=1).routing is None
