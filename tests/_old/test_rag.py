import unittest

from src.rag.chunker import chunk_documents
from src.rag.generator import ExtractiveGenerator
from src.rag.vector_store import VectorStore


class RagTests(unittest.TestCase):
    def test_chunking_overlap_and_vector_search(self):
        docs = [{"doc_id": "d1", "text": "بيع عقار. " * 20}, {"doc_id": "d2", "text": "نزاع كراء."}]
        chunks = chunk_documents(docs, chunk_size=35, overlap=5)
        self.assertGreater(len(chunks), 2)
        store = VectorStore(chunks)
        self.assertEqual(store.search("بيع عقار")[0]["doc_id"], "d1")

    def test_generator_uses_only_matching_source_sentences(self):
        result = ExtractiveGenerator().generate(
            "ما موضوع عقد البيع؟",
            [{"doc_id": "d1", "text": "موضوع القرار عقد البيع. لا توجد معلومات أخرى."}],
        )
        self.assertIn("عقد البيع", result["answer"])
        self.assertEqual(result["sources"], ["d1"])


if __name__ == "__main__":
    unittest.main()
