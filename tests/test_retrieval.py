import json

import numpy as np
import pytest

from nl2sql.config import Config
from nl2sql.distractors import distractors
from nl2sql.embeddings import normalize
from nl2sql.examples import load_examples
from nl2sql.fk_graph import expand
from nl2sql.retrieval import DenseIndex, Retriever, rrf
from shared.interfaces import Document, Hit


class FakeEmbedder:
    settings = {"model": "test", "revision": "1", "normalized": True}

    def encode(self, texts, query=False):
        return normalize([[1.0, len(t) % 5 + 1] for t in texts])


def test_normalization():
    assert np.allclose(np.linalg.norm(normalize([[3, 4], [1, 2]]), axis=1), 1)
    with pytest.raises(ValueError):
        normalize([[0, 0]])


def test_dense_roundtrip_and_invalidation(tmp_path):
    docs = [Document("a", "hello"), Document("b", "world!")]
    e = FakeEmbedder()
    index = DenseIndex(docs, e)
    assert index.index.metric_type == 0
    index.save(tmp_path)
    loaded = DenseIndex.load(tmp_path, docs, e)
    assert [h.document.id for h in loaded.search("hello", 2)] == [
        h.document.id for h in index.search("hello", 2)
    ]
    with pytest.raises(ValueError):
        DenseIndex.load(tmp_path, list(reversed(docs)), e)
    e.settings = {"model": "changed"}
    with pytest.raises(ValueError):
        DenseIndex.load(tmp_path, docs, e)


def test_rrf_and_determinism():
    a, b = Document("a", "a"), Document("b", "b")
    fused = rrf(
        [
            [Hit(a, 1, "dense"), Hit(b, 0.5, "dense")],
            [Hit(b, 1, "bm25"), Hit(a, 0.5, "bm25")],
        ]
    )
    assert [h.document.id for h in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_fk_bridge():
    r = expand(["Customer", "Track"])
    assert {"Invoice", "InvoiceLine"} <= set(r["bridge_tables"])
    assert "Invoice.CustomerId" in r["join_columns"]
    assert expand(["Customer", "Nonexistent"])["disconnected"]


def test_no_eval_leakage():
    from nl2sql.config import ROOT

    pool = load_examples()
    gold = json.loads((ROOT / "data/evaluation.json").read_text())
    dev = json.loads((ROOT / "data/development.json").read_text())
    assert len(pool) == 30 and len(gold) == 60
    for q in gold + dev:
        assert all(q["id"] != d.id and q["question"] not in d.text for d in pool)
    assert len({q["question"].lower() for q in gold}) == 60
    assert {
        d: sum(q["difficulty"] == d for q in gold) for d in ("easy", "medium", "hard")
    } == {"easy": 20, "medium": 25, "hard": 15}


def test_distractors():
    assert distractors(100) == distractors(100)
    assert len(distractors(25)) == 25
    from nl2sql.db import schema

    assert not {d.table for d in distractors(100)} & set(schema())


def test_langchain_bm25_parity():
    from lc.retrievers import LangChainRetriever

    docs = load_examples()
    cfg = Config(retrieval="bm25")
    e = FakeEmbedder()
    a = Retriever(docs, e, cfg, "test").search("album tracks", 5)
    b = LangChainRetriever(docs, e, cfg, "test").search("album tracks", 5)
    assert [(h.document.id, h.score) for h in a] == [
        (h.document.id, h.score) for h in b
    ]


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_langchain_dense_parity(mode, tmp_path, monkeypatch):
    import nl2sql.retrieval as module
    from lc.retrievers import LangChainRetriever

    monkeypatch.setattr(module, "ROOT", tmp_path)
    docs = load_examples()
    cfg = Config(retrieval=mode)
    e = FakeEmbedder()
    a = Retriever(docs, e, cfg, "test").search("album tracks", 5)
    b = LangChainRetriever(docs, e, cfg, "test").search("album tracks", 5)
    assert [h.document.id for h in a] == [h.document.id for h in b]
    assert np.allclose([h.score for h in a], [h.score for h in b])


def test_bge_encoding_convention(monkeypatch):
    import nl2sql.embeddings as module
    from nl2sql.config import BGE, BGE_QUERY_PREFIX

    seen = []

    class Local:
        def encode(self, texts, **kwargs):
            seen.append(texts)
            return [[3.0, 4.0] for _ in texts]

    monkeypatch.setattr(module, "local_model", lambda *args: Local())
    embedder = module.Embedder(BGE)
    embedder.encode(["question"], query=True)
    embedder.encode(["document"])
    assert seen == [[BGE_QUERY_PREFIX + "question"], ["document"]]


def test_reranking(monkeypatch):
    import nl2sql.rerank as module

    class Fake:
        def predict(self, pairs):
            return [0.1, 0.9]

    monkeypatch.setattr(module, "model", lambda: Fake())
    hits = module.rerank(
        "question",
        [Hit(Document("a", "a"), 1, "dense"), Hit(Document("b", "b"), 0, "dense")],
    )
    assert [h.document.id for h in hits] == ["b", "a"]
