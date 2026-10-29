"""Simple transparent metrics for retrieval coverage and answer grounding."""

from collections.abc import Iterable


def context_recall(answer_terms: Iterable[str], context_terms: Iterable[str]) -> float:
    answer, context = set(answer_terms), set(context_terms)
    return len(answer & context) / len(answer) if answer else 0.0


def faithfulness(answer_terms: Iterable[str], context_terms: Iterable[str]) -> float:
    return context_recall(answer_terms, context_terms)
