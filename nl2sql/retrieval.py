import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from nl2sql.config import ROOT, db_checksum, digest
from nl2sql.embeddings import normalize
from shared.interfaces import Hit


def tokenize(text):
    return re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())


def rrf(rankings, k=60):
    scores = {}
    docs = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            key = hit.document.id
            docs[key] = hit.document
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
    return [
        Hit(docs[i], s, "rrf")
        for i, s in sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    ]


class DenseIndex:
    def __init__(self, docs, embedder, vectors=None):
        import faiss

        self.docs = docs
        self.embedder = embedder
        self.vectors = normalize(
            vectors if vectors is not None else embedder.encode([d.text for d in docs])
        )
        if len(docs) != len(self.vectors):
            raise ValueError("Position alignment mismatch")
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    def manifest(self):
        return {
            "database_sha256": db_checksum(),
            "documents_sha256": digest([asdict(d) for d in self.docs]),
            "embedding": self.embedder.settings,
            "dimensions": self.vectors.shape[1],
            "count": len(self.docs),
            "format": 1,
        }

    def search(self, question, k):
        q = normalize(self.embedder.encode([question], query=True))
        # Search entire tiny corpus then impose ID tie breaking for parity.
        scores, positions = self.index.search(q, len(self.docs))
        hits = [
            Hit(self.docs[int(i)], float(s), "dense")
            for s, i in zip(scores[0], positions[0])
            if i >= 0
        ]
        return sorted(hits, key=lambda h: (-h.score, h.document.id))[:k]

    def save(self, path):
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "metadata.json").write_text(
            json.dumps([asdict(d) for d in self.docs], ensure_ascii=False),
            encoding="utf-8",
        )
        manifest = self.manifest()
        manifest["index_sha256"] = hashlib.sha256(
            (path / "index.faiss").read_bytes()
        ).hexdigest()
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path, docs, embedder):
        import faiss

        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        for key, value in {
            "database_sha256": db_checksum(),
            "documents_sha256": digest([asdict(d) for d in docs]),
            "embedding": embedder.settings,
            "count": len(docs),
            "format": 1,
        }.items():
            if manifest.get(key) != value:
                raise ValueError("Stale/incompatible index: " + key)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata != [asdict(d) for d in docs]:
            raise ValueError("Metadata alignment mismatch")
        if (
            hashlib.sha256((path / "index.faiss").read_bytes()).hexdigest()
            != manifest["index_sha256"]
        ):
            raise ValueError("Index checksum mismatch")
        # Only load trusted, locally built repository assets; FAISS is a native binary format.
        idx = faiss.read_index(str(path / "index.faiss"))
        if (
            not isinstance(idx, faiss.IndexFlatIP)
            or idx.d != manifest["dimensions"]
            or idx.ntotal != len(docs)
        ):
            raise ValueError("Index shape/type mismatch")
        obj = cls.__new__(cls)
        obj.docs = docs
        obj.embedder = embedder
        obj.index = idx
        obj.vectors = idx.reconstruct_n(0, idx.ntotal)
        if not np.allclose(np.linalg.norm(obj.vectors, axis=1), 1, atol=1e-5):
            raise ValueError("Index not normalized")
        return obj


class Retriever:
    def __init__(self, docs, embedder, config, namespace):
        self.docs = docs
        self.config = config
        self.bm25 = BM25Okapi([tokenize(d.text) for d in docs])
        self.dense = None
        if config.retrieval != "bm25":
            path = (
                ROOT
                / "indexes"
                / digest(
                    {
                        "docs": [asdict(d) for d in docs],
                        "embedding": embedder.settings,
                        "namespace": namespace,
                    }
                )[:20]
            )
            if path.exists():
                self.dense = DenseIndex.load(path, docs, embedder)
            else:
                self.dense = DenseIndex(docs, embedder)
                self.dense.save(path)

    def search(self, question, k):
        if k == 0:
            return []
        c = self.config
        scores = self.bm25.get_scores(tokenize(question))
        sparse = sorted(
            [Hit(d, float(s), "bm25") for d, s in zip(self.docs, scores)],
            key=lambda h: (-h.score, h.document.id),
        )[: c.candidates]
        dense = self.dense.search(question, c.candidates) if self.dense else []
        hits = (
            dense
            if c.retrieval == "dense"
            else sparse
            if c.retrieval == "bm25"
            else rrf([dense, sparse], c.rrf_k)
        )

        def compact(values):
            return [
                {"id": h.document.id, "score": h.score, "stage": h.stage}
                for h in values
            ]

        self.trace = {
            "dense_candidates": compact(dense),
            "bm25_candidates": compact(sparse),
            "fused_or_selected": compact(hits),
            "candidate_budget": c.candidates,
            "final_budget": k,
        }
        if c.rerank:
            from nl2sql.rerank import rerank

            hits = rerank(question, hits[: c.candidates])
        self.trace["reranked"] = compact(hits) if c.rerank else None
        self.trace["final"] = compact(hits[:k])
        return hits[:k]
