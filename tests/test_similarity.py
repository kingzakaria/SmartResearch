import unittest

from src.evaluation.search_metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from src.similarity.levenshtein import levenshtein_distance
from src.similarity.measures import cosine_similarity, dice_similarity, jaccard_similarity


class SimilarityTests(unittest.TestCase):
    def test_sparse_cosine_and_set_metrics(self):
        self.assertAlmostEqual(cosine_similarity({"a": 1.0}, {"a": 1.0, "b": 1.0}), 2**-0.5)
        self.assertEqual(jaccard_similarity(["a", "b"], ["b", "c"]), 1 / 3)
        self.assertEqual(dice_similarity(["a", "b"], ["b", "c"]), 0.5)
        self.assertEqual(cosine_similarity({}, {}), 0.0)

    def test_levenshtein_and_ranked_metrics(self):
        self.assertEqual(levenshtein_distance("كتاب", "كتابة"), 1)
        ranked, relevant = ["a", "x", "b"], {"a", "b"}
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(recall_at_k(ranked, relevant, 3), 1.0)
        self.assertAlmostEqual(average_precision(ranked, relevant), (1 + 2 / 3) / 2)
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], relevant, 2), 1.0)


if __name__ == "__main__":
    unittest.main()
