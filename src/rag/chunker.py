"""Step 1 of RAG: cut each decision into passages ("chunks").

An LLM cannot read 358 decisions, and one decision is too long and too mixed to be a good unit for retrieval.
A passage of about 400 words is short enough to be about one idea and long enough to make sense alone.
Consecutive passages overlap (80 words) so that an idea cut at a border appears whole in one of them.

The cut prefers the end of a sentence: if a full stop is found in the last part of the window, the passage ends there.
"""
from ..search.highlight import reading_text


def split_words(text: str, size: int = 400, overlap: int = 80, look_back: int = 60) -> list[str]:
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while True:
        end = min(start + size, len(words))
        if end < len(words):                                  # not the last passage: try to end on a sentence
            for i in range(end, max(start + size - look_back, start + 1), -1):
                if words[i - 1].endswith((".", "؟", "!")):
                    end = i
                    break
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            return chunks
        start = max(end - overlap, start + 1)                 # step back by the overlap, always moving forward


def build_chunks(documents, skip_ids=(), size: int = 400, overlap: int = 80) -> list[dict]:
    """[{chunk_id, doc_id, n, text}] for every decision, except the duplicates in `skip_ids`
    (the same text three times would fill the answer with copies)."""
    skip = {str(i) for i in skip_ids}
    docs = documents.values() if isinstance(documents, dict) else documents
    chunks = []
    for doc in docs:
        doc_id = str(doc["doc_id"])
        if doc_id in skip:
            continue
        text = reading_text(doc.get("body", ""), doc.get("subject"))
        for n, piece in enumerate(split_words(text, size, overlap)):
            chunks.append({"chunk_id": f"{doc_id}#{n}", "doc_id": doc_id, "n": n, "text": piece})
    return chunks
