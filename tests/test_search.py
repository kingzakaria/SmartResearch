import unittest

from src.morphology.analyzer import ArabicMorphologyAnalyzer
from src.search.engine import SearchEngine


class SearchTests(unittest.TestCase):
    def test_query_returns_relevant_document(self):
        docs = [
            {"doc_id": "sale", "text": "عقد بيع عقار وتسجيل الملكية"},
            {"doc_id": "rent", "text": "نزاع كراء وإفراغ محل تجاري"},
        ]
        engine = SearchEngine(docs, ArabicMorphologyAnalyzer(stemming=None))
        results = engine.search("عقد بيع عقار")
        self.assertEqual(results[0]["doc_id"], "sale")
        self.assertEqual(engine.search("غير موجود"), [])


if __name__ == "__main__":
    unittest.main()
