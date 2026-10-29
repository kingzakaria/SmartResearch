"""Grounding instructions for optional external answer-generation adapters."""

SYSTEM_PROMPT = (
    "Answer only using the supplied context. If the context does not contain "
    "the answer, say that the available documents do not provide it. Cite source IDs."
)


def build_prompt(question: str, contexts: list[dict[str, str]]) -> str:
    passages = "\n\n".join(f"[{item['source']}]\n{item['text']}" for item in contexts)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{passages}\n\nQuestion: {question}\nAnswer:"
