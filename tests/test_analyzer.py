import unittest

from src.morphology.analyzer import ArabicMorphologyAnalyzer


class AnalyzerTests(unittest.TestCase):
    def test_removes_normalized_stopwords_and_stems(self):
        analyzer = ArabicMorphologyAnalyzer(stemming="light", extra_stopwords=["المحكمة"])
        self.assertEqual(analyzer.analyze("المحكمة والعقارات"), ["عقار"])

    def test_can_disable_stemming(self):
        analyzer = ArabicMorphologyAnalyzer(stemming=None)
        self.assertEqual(analyzer.analyze("العقارات"), ["العقارات"])


if __name__ == "__main__":
    unittest.main()
