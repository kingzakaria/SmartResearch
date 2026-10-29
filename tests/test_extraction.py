import tempfile
import unittest
from pathlib import Path

from src.extraction.loader import load_documents


class ExtractionTests(unittest.TestCase):
    def test_loads_utf8_and_extracts_metadata_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.txt"
            original = "قرار رقم: ١٢٣ بتاريخ 03/04/2024\nملف رقم 45/2024\nالموضوع: بيع عقار\n\nنص الحكم"
            path.write_text(original, encoding="utf-8")
            documents = load_documents(directory)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["doc_id"], "decision")
            self.assertEqual(documents[0]["metadata"]["decision_number"], "١٢٣")
            self.assertEqual(documents[0]["metadata"]["date"], "03/04/2024")
            self.assertEqual(documents[0]["metadata"]["case_number"], "45/2024")


if __name__ == "__main__":
    unittest.main()
