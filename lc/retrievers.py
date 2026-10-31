"""LangChain retrieval components with shared encoding and fusion semantics."""

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from nl2sql.retrieval import Retriever, rrf, tokenize
from shared.interfaces import Hit


class Adapter(Embeddings):
    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts):
        return self.embedder.encode(texts).tolist()

    def embed_query(self, text):
        return self.embedder.encode([text], query=True)[0].tolist()


class LangChainRetriever(Retriever):
    def __init__(self, docs, embedder, config, namespace):
        super().__init__(docs, embedder, config, namespace)
        self.lookup = {d.id: d for d in docs}
        lc_docs = [LCDocument(page_content=d.text, metadata={"id": d.id}) for d in docs]
        self.sparse = BM25Retriever.from_documents(
            lc_docs, preprocess_func=tokenize, k=len(docs)
        )
        if self.dense:
            self.store = FAISS(
                Adapter(embedder),
                self.dense.index,
                InMemoryDocstore({d.metadata["id"]: d for d in lc_docs}),
                {i: d.id for i, d in enumerate(docs)},
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )

    def search(self, question, k):
        if k == 0:
            return []
        c = self.config
        # BM25Retriever uses NumPy tie ordering; recompute scores for shared stable ordering.
        sparse_docs = self.sparse.invoke(question)
        all_scores = self.sparse.vectorizer.get_scores(tokenize(question))
        scores = {d.id: float(s) for d, s in zip(self.docs, all_scores)}
        sparse = sorted(
            [
                Hit(self.lookup[d.metadata["id"]], scores[d.metadata["id"]], "bm25")
                for d in sparse_docs
            ],
            key=lambda h: (-h.score, h.document.id),
        )[: c.candidates]
        dense = []
        if self.dense:
            dense = sorted(
                [
                    Hit(self.lookup[d.metadata["id"]], float(s), "dense")
                    for d, s in self.store.similarity_search_with_score(
                        question, k=len(self.docs)
                    )
                ],
                key=lambda h: (-h.score, h.document.id),
            )[: c.candidates]
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
