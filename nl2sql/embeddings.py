from functools import lru_cache

import numpy as np

from nl2sql.config import (
    BGE,
    BGE_QUERY_PREFIX,
    EMBEDDING_BATCH_SIZE,
    GEMINI_EMBED,
    MINILM,
    REVISIONS,
)


def normalize(vectors):
    x = np.asarray(vectors, dtype="float32")
    x = np.atleast_2d(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (norms == 0).any():
        raise ValueError("Nonfinite or zero embeddings")
    return np.ascontiguousarray(x / norms)


@lru_cache(maxsize=1)
def local_model(name, revision):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        name, revision=revision, device="cpu", trust_remote_code=False
    )


class Embedder:
    def __init__(self, name=MINILM, budget=None):
        if name not in (BGE, MINILM, GEMINI_EMBED):
            raise ValueError("Unsupported embedding")
        self.name = name
        self.budget = budget

    @property
    def settings(self):
        return {
            "model": self.name,
            "revision": REVISIONS[self.name],
            "normalized": True,
            "query_prefix": BGE_QUERY_PREFIX if self.name == BGE else "",
            "document_prefix": "",
            "gemini_tasks": self.name == GEMINI_EMBED,
        }

    def encode(self, texts, query=False):
        if self.name == GEMINI_EMBED:
            from google import genai

            from nl2sql.llm import get_api_key

            if self.budget is None:
                raise ValueError("Gemini embeddings require an explicit shared budget")
            from nl2sql.llm import ModelError

            client = genai.Client(
                api_key=get_api_key(), http_options={"retry_options": {"attempts": 1}}
            )
            vectors = []
            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[start : start + EMBEDDING_BATCH_SIZE]
                reserved = sum(len(t.encode("utf-8")) for t in batch)
                self.budget.consume(reserved)
                try:
                    response = client.models.embed_content(
                        model=self.name,
                        contents=batch,
                        config={
                            "task_type": "RETRIEVAL_QUERY"
                            if query
                            else "RETRIEVAL_DOCUMENT"
                        },
                    )
                except Exception as exc:
                    raise ModelError(
                        "Embedding request failed ("
                        + type(exc).__name__
                        + "). Check key, quota or connection."
                    ) from None
                vectors.extend(e.values for e in response.embeddings)
            return normalize(vectors)
        prefix = BGE_QUERY_PREFIX if self.name == BGE and query else ""
        return normalize(
            local_model(self.name, REVISIONS[self.name]).encode(
                [prefix + t for t in texts],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
