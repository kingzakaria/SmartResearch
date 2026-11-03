"""RAG on a tiny corpus with a fake embedder and a fake LLM (no network). Run: pytest tests/test_rag.py"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.morphology.analyzer import analyzer_from_config  # noqa: E402
from src.rag.answer import RagService, build_prompt, cited_numbers  # noqa: E402
from src.rag.chunker import build_chunks, split_words  # noqa: E402
from src.rag.gemini import GeminiClient, GeminiError, load_api_key  # noqa: E402
from src.rag.retriever import HybridRetriever, fuse  # noqa: E402

ANALYZER = analyzer_from_config(config, "light")


def make_chunks(texts):
    return [{"chunk_id": cid, "doc_id": cid.split("#")[0], "text": t, "terms": ANALYZER.analyze(t)}
            for cid, t in texts.items()]


TEXTS = {
    "1#0": "دعوى الشفعة وبيع العقار بالمزاد",
    "1#1": "الشفعة وتزاحم الشفعاء في بيع العقار",
    "1#2": "الشفعة شروطها الاخطار بالبيع",
    "2#0": "مطلب التحفيظ والتعرض على مطلب التحفيظ العقاري",
    "3#0": "دعوى الحيازة شروط الحيازة الهادئة",
}


class FakeEmbedder:
    """Vectors on the unit circle chosen by hand: the question is closest to 3#0."""
    def __init__(self, fail=False):
        self.fail = fail

    def embed_query(self, text):
        if self.fail:
            raise GeminiError("quota", 429)
        return np.array([0.0, 1.0], dtype=np.float32)


EMB_IDS = list(TEXTS)
EMB = np.array([[1, 0], [1, 0], [1, 0], [0.7, 0.7], [0, 1]], dtype=np.float32)


# ---------- chunker ----------
def test_split_words_overlap_and_coverage():
    words = [f"w{i}" for i in range(1000)]
    chunks = split_words(" ".join(words), size=400, overlap=80)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 400 for c in chunks)
    assert chunks[1].split()[:80] == chunks[0].split()[-80:]            # consecutive passages overlap
    assert set(" ".join(chunks).split()) == set(words)                   # nothing is lost


def test_split_words_prefers_sentence_end_and_short_text_is_one_chunk():
    text = " ".join(["كلمة"] * 380 + ["نهاية."] + ["كلمة"] * 100)
    first = split_words(text, 400, 80)[0]
    assert first.endswith("نهاية.")
    assert split_words("نص قصير", 400, 80) == ["نص قصير"]
    assert split_words("", 400, 80) == []
    with pytest.raises(ValueError):
        split_words("a b", 10, 10)


def test_build_chunks_skips_duplicates_and_subject_line():
    docs = [{"doc_id": 1, "body": "الموضوع الشفعة\nنص القرار الاول", "subject": "الموضوع الشفعة"},
            {"doc_id": 2, "body": "نص القرار الثاني"}]
    chunks = build_chunks(docs, skip_ids=[2])
    assert [c["chunk_id"] for c in chunks] == ["1#0"]
    assert "الموضوع" not in chunks[0]["text"]


# ---------- retriever ----------
def test_fuse_rewards_agreement():
    scores = fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62) and scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] > scores["d"] and scores["b"] > scores["c"]


def test_keyword_only_retrieval():
    r = HybridRetriever(ANALYZER, make_chunks(TEXTS), max_per_doc=2)
    assert not r.hybrid
    passages = r.retrieve("الشفعة", top=5)
    assert {p.doc_id for p in passages} == {"1"}
    assert all(p.embedding_rank is None and p.keyword_rank for p in passages)
    assert [p.n for p in passages] == [1, 2]                              # capped at 2 passages per decision


def test_hybrid_brings_a_passage_with_no_shared_word():
    r = HybridRetriever(ANALYZER, make_chunks(TEXTS), (EMB_IDS, EMB), FakeEmbedder(), max_per_doc=1)
    assert r.hybrid
    passages = r.retrieve("الشفعة", top=3)
    ids = [p.chunk_id for p in passages]
    assert "3#0" in ids and "2#0" in ids                                  # found by meaning only
    assert len({p.doc_id for p in passages}) == len(passages)             # one passage per decision here
    assert next(p for p in passages if p.chunk_id == "3#0").keyword_rank is None


def test_embedding_failure_falls_back_to_keywords():
    r = HybridRetriever(ANALYZER, make_chunks(TEXTS), (EMB_IDS, EMB), FakeEmbedder(fail=True))
    passages = r.retrieve("الشفعة")
    assert passages and r.last_warning and "keyword" in r.last_warning


def test_embeddings_for_other_passages_are_ignored():
    r = HybridRetriever(ANALYZER, make_chunks(TEXTS), (["9#0"], np.ones((1, 2), dtype=np.float32)), FakeEmbedder())
    assert not r.hybrid


def test_no_match_returns_nothing():
    r = HybridRetriever(ANALYZER, make_chunks(TEXTS))
    assert r.retrieve("zzz qqq") == []


# ---------- answer ----------
class FakeLLM:
    def __init__(self, text="الشفعة حق قانوني [1][2]."):
        self.text, self.calls = text, []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return self.text


DOCS = {"1": {"decision_no": "1/10", "date": "2023-05-01", "case_no": "55/2022", "subject": "الشفعة"},
        "2": {"decision_no": "2/11"}, "3": {}}


def test_cited_numbers():
    assert cited_numbers("نص [1] و [2, 3] و[4][5] وليس (6) ولا [x]") == {1, 2, 3, 4, 5}


def test_ask_returns_answer_sources_and_marks_cited():
    retriever = HybridRetriever(ANALYZER, make_chunks(TEXTS), max_per_doc=2)
    llm = FakeLLM()
    result = RagService(retriever, llm, DOCS).ask("ما هي الشفعة؟")
    assert result["answer"].startswith("الشفعة") and result["retrieval"] == "keyword"
    assert [s["n"] for s in result["sources"]] == [1, 2] and all(s["cited"] for s in result["sources"])
    system, prompt = llm.calls[0]
    assert "ONLY" in system and "[1] (قرار عدد 1/10" in prompt and "ما هي الشفعة؟" in prompt


def test_ask_does_not_call_the_llm_without_passages():
    llm = FakeLLM()
    result = RagService(HybridRetriever(ANALYZER, make_chunks(TEXTS)), llm, DOCS).ask("zzz qqq")
    assert result["sources"] == [] and llm.calls == []


def test_prompt_numbers_the_passages():
    retriever = HybridRetriever(ANALYZER, make_chunks(TEXTS))
    prompt = build_prompt("سؤال", retriever.retrieve("الشفعة"), DOCS)
    assert prompt.index("[1]") < prompt.index("[2]") < prompt.index("Question:")


# ---------- gemini client (fake HTTP) ----------
class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._p, self.text = status, payload, str(payload)

    def json(self):
        return self._p


class FakeHttp:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def post(self, url, json=None, headers=None):
        self.requests.append((url, json, headers))
        return self.responses.pop(0)


def test_embed_normalises_and_retries_on_429():
    http = FakeHttp([FakeResponse(429, {"error": {"message": "slow down"}}),
                     FakeResponse(200, {"embeddings": [{"values": [3, 4]}, {"values": [0, 2]}]})])
    pauses = []
    client = GeminiClient("KEY", config, http=http, sleep=pauses.append)
    vectors = client.embed(["a", "b"])
    assert pauses == [5] and len(http.requests) == 2
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1) and np.allclose(vectors[0], [0.6, 0.8])
    url, body, headers = http.requests[0]
    assert url.endswith(":batchEmbedContents") and headers["x-goog-api-key"] == "KEY"
    assert body["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"


def test_generate_and_errors():
    ok = FakeHttp([FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "مرحبا"}]}}]})])
    assert GeminiClient("K", config, http=ok).generate("sys", "hi") == "مرحبا"
    blocked = FakeHttp([FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}})])
    with pytest.raises(GeminiError, match="SAFETY"):
        GeminiClient("K", config, http=blocked).generate("s", "p")
    bad = FakeHttp([FakeResponse(400, {"error": {"message": "API key not valid"}})])
    with pytest.raises(GeminiError, match="not valid") as info:
        GeminiClient("K", config, http=bad).generate("s", "p")
    assert info.value.status == 400


def test_load_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('OTHER=1\nGEMINI_API_KEY="abc123"\n', encoding="utf-8")
    cfg = type("C", (), {"ENV_FILE": env})
    assert load_api_key(cfg) == "abc123"
    monkeypatch.setenv("GEMINI_API_KEY", "fromenv")
    assert load_api_key(cfg) == "fromenv"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert load_api_key(type("C", (), {"ENV_FILE": tmp_path / "missing"})) is None
