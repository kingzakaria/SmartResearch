"""Split documents into overlapping character chunks."""

from collections.abc import Iterable, Mapping

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def chunk_documents(
    documents: Iterable[Mapping[str, object]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, object]]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    chunks = []
    for document in documents:
        text = str(document.get("text", "")).strip()
        doc_id = str(document["doc_id"])
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary
            content = text[start:end].strip()
            if content:
                chunks.append({"chunk_id": f"{doc_id}:{len(chunks)}", "doc_id": doc_id, "text": content})
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
