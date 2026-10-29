import math
import unittest

from src.indexing.tfidf import TfidfIndex


class TfidfTests(unittest.TestCase):
    def test_uses_base_ten_idf_and_normalizes_vectors(self):
        index = TfidfIndex().fit(
            {"d1": ["rare", "common"], "d2": ["common", "other"], "d3": ["common", "other"]}
        )
        self.assertEqual(index.document_frequency["common"], 3)
        expected = 0.5 * math.log10(3)
        self.assertAlmostEqual(index.document_frequency["rare"], 1)
        self.assertAlmostEqual(index.vectors["d1"]["rare"], expected)
        self.assertEqual(index.transform(["unknown"]), {})
        self.assertAlmostEqual(index.vectors["d2"]["other"], 0.5 * math.log10(3 / 2))
        self.assertAlmostEqual(expected, 0.2385606274, places=8)

    def test_empty_corpus_and_documents(self):
        self.assertEqual(TfidfIndex().fit({}).transform(["term"]), {})
        self.assertEqual(TfidfIndex().fit({"empty": []}).vectors, {"empty": {}})


if __name__ == "__main__":
    unittest.main()
