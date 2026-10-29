import unittest

from src.morphology.normalizer import normalize_arabic


class NormalizerTests(unittest.TestCase):
    def test_removes_diacritics_and_tatweel(self):
        self.assertEqual(normalize_arabic("إِنَّ اللّٰهَ"), "ان الله")
        self.assertEqual(normalize_arabic("مـحـمـد"), "محمد")

    def test_normalizes_alef_and_alif_maqsura(self):
        self.assertEqual(normalize_arabic("أحمد، آلى"), "احمد، الي")


if __name__ == "__main__":
    unittest.main()
