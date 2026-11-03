"""Google Gemini over plain HTTPS (httpx): the embedding model and the text model, one API key for both.

Why not a library: two endpoints are enough, and it keeps the install light on Windows.
The key comes from the environment variable GEMINI_API_KEY or from the file .env at the project root.
"""
import os
import time

import httpx
import numpy as np


class GeminiError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def load_api_key(config) -> str | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    path = getattr(config, "ENV_FILE", None)
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GEMINI_API_KEY":
                return value.strip().strip("\"'") or None
    return None


class GeminiClient:
    def __init__(self, api_key: str, config, http: httpx.Client | None = None, sleep=time.sleep):
        self.api_key = api_key
        self.base = config.GEMINI_API_BASE
        self.llm_model = config.GEMINI_LLM_MODEL
        self.embed_model = config.GEMINI_EMBED_MODEL
        self.dim = config.EMBED_DIM
        timeout = getattr(config, "GEMINI_TIMEOUT", 180)           # seconds to wait for an answer (slow networks, slow models)
        self.http = http or httpx.Client(timeout=httpx.Timeout(timeout, connect=20))
        self.sleep = sleep

    # ---------- plumbing ----------
    def _post(self, path: str, body: dict, retries: int = 0) -> dict:
        """POST with a pause and a new try when the free tier says "too many requests" (429) or is busy (503)."""
        for attempt in range(retries + 1):
            try:
                response = self.http.post(f"{self.base}/{path}", json=body, headers={"x-goog-api-key": self.api_key})
            except httpx.HTTPError as error:                  # connection reset, timeout...: wait and try again
                if attempt < retries:
                    self.sleep(min(60, 5 * 2 ** attempt))
                    continue
                raise GeminiError(f"Cannot reach Gemini: {error}") from error
            if response.status_code in (429, 503) and attempt < retries:
                self.sleep(min(60, 5 * 2 ** attempt))
                continue
            if response.status_code != 200:
                try:
                    message = response.json()["error"]["message"]
                except Exception:
                    message = response.text[:200]
                raise GeminiError(message, response.status_code)
            return response.json()
        raise GeminiError("unreachable")

    # ---------- embeddings ----------
    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT", retries: int = 6) -> np.ndarray:
        """One unit-length vector per text. Documents and queries use different `task` values (the model is
        trained to put a question near the passages that answer it). With fewer than 3072 dimensions Google
        does not normalise the vectors, so we do it: the dot product of two of them is then their cosine."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        body = {"requests": [{"model": f"models/{self.embed_model}", "content": {"parts": [{"text": t}]},
                              "taskType": task, "outputDimensionality": self.dim} for t in texts]}
        data = self._post(f"models/{self.embed_model}:batchEmbedContents", body, retries)
        vectors = np.array([e["values"] for e in data["embeddings"]], dtype=np.float32)
        if len(vectors) != len(texts):
            raise GeminiError("Gemini returned a different number of embeddings than texts")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], "RETRIEVAL_QUERY", retries=2)[0]

    # ---------- text generation ----------
    def generate(self, system: str, prompt: str, temperature: float = 0.2) -> str:
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature}}
        data = self._post(f"models/{self.llm_model}:generateContent", body, retries=5)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError):
            reason = data.get("promptFeedback", {}).get("blockReason") or "empty answer"
            raise GeminiError(f"Gemini gave no answer ({reason})")
