"""Grounding instructions for optional external answer-generation adapters."""

SYSTEM_PROMPT = (
    "Answer in the same language as the question, using only the supplied context. "
    "Do not infer facts that the context does not state. If the context does not "
    "contain the answer, say that the available documents do not provide it. "
    "Cite supporting passages using their source IDs in square brackets, such as [123]."
)


def build_prompt(question: str, contexts: list[dict[str, str]]) -> str:
    passages = "\n\n".join(f"[{item['source']}]\n{item['text']}" for item in contexts)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{passages}\n\nQuestion: {question}\nAnswer:"
