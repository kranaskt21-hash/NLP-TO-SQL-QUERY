from functools import lru_cache

from nl2sql.config import RERANKER, REVISIONS
from shared.interfaces import Hit


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        RERANKER,
        device="cpu",
        revision=REVISIONS.get(RERANKER, "main"),
        trust_remote_code=False,
    )


def rerank(question, hits):
    if not hits:
        return []
    scores = model().predict([(question, h.document.text) for h in hits])
    return sorted(
        [Hit(h.document, float(s), "cross-encoder") for h, s in zip(hits, scores)],
        key=lambda h: (-h.score, h.document.id),
    )
