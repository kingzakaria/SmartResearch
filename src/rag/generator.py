"""Extractive answer generator that cannot introduce unsupported facts."""

import re
from collections.abc import Mapping


class ExtractiveGenerator:
    def generate(self, question: str, passages: list[Mapping[str, object]]) -> dict[str, object]:
        terms = set(re.findall(r"[^\W_]+", question.casefold(), flags=re.UNICODE))
        candidates = []
        sources = []
        for passage in passages:
            text = str(passage.get("text", ""))
            source = str(passage.get("doc_id", "unknown"))
            for sentence in re.split(r"(?<=[.!؟?])\s+|\n+", text):
                overlap = len(terms & set(re.findall(r"[^\W_]+", sentence.casefold(), flags=re.UNICODE)))
                if overlap:
                    candidates.append((overlap, sentence.strip(), source))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        answer_sentences = []
        for _, sentence, source in candidates[:3]:
            answer_sentences.append(sentence)
            if source not in sources:
                sources.append(source)
        answer = " ".join(answer_sentences) if answer_sentences else "لا تتضمن المقاطع المسترجعة معلومات كافية للإجابة."
        return {"answer": answer, "sources": sources}
