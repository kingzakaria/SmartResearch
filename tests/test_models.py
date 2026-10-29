import unittest

from src.classification.evaluate import evaluate_classifier
from src.classification.train import train_classifier
from src.clustering.kmeans import cluster_documents
from src.indexing.bm25 import BM25Index


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "a1": ["بيع", "عقار", "ملكية"],
            "a2": ["بيع", "أرض", "ملكية"],
            "b1": ["كراء", "محل", "إفراغ"],
            "b2": ["كراء", "عقار", "إفراغ"],
        }
        self.labels = {"a1": "بيع", "a2": "بيع", "b1": "كراء", "b2": "كراء"}

    def test_clustering_and_knn_cross_validation(self):
        clusters = cluster_documents(self.documents, n_clusters=2)
        self.assertEqual(set(clusters), set(self.documents))
        model = train_classifier(self.documents, self.labels, algorithm="knn")
        metrics = evaluate_classifier(model, self.documents, self.labels, folds=2)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_bm25_reuses_query_terms_across_documents(self):
        scores = BM25Index({"a": ["بيع"], "b": ["كراء"]}).score(term for term in ["بيع"])
        self.assertGreater(scores["a"], 0)
        self.assertEqual(scores["b"], 0)


if __name__ == "__main__":
    unittest.main()
