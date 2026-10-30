import unittest

from src.morphology.tokenizer import tokenize


class TokenizerTests(unittest.TestCase):
    def test_keeps_arabic_words_and_numbers(self):
        self.assertEqual(tokenize("قرار رقم ١٢٣، ملف 45/2024!"), ["قرار", "رقم", "١٢٣", "ملف", "45", "2024"])

    def test_normalizes_before_tokenizing(self):
        self.assertEqual(tokenize("إِبْرَاهِيم؛ ٢"), ["ابراهيم", "٢"])


if __name__ == "__main__":
    unittest.main()
