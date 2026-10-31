import argparse

from nl2sql.config import MINILM, Config
from nl2sql.distractors import distractors
from nl2sql.embeddings import Embedder
from nl2sql.examples import load_examples
from nl2sql.retrieval import Retriever
from nl2sql.schema_docs import documents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding", default=MINILM)
    p.add_argument("--call-budget", type=int, default=4)
    p.add_argument("--token-budget", type=int, default=100000)
    p.add_argument("--distractors", type=int, default=0)
    a = p.parse_args()
    config = Config(embedding=a.embedding, distractors=a.distractors)
    from nl2sql.llm import Budget

    embedder = Embedder(a.embedding, Budget(a.call_budget, a.token_budget))
    for namespace, docs in [
        ("schema", documents() + distractors(a.distractors)),
        ("examples", load_examples()),
    ]:
        r = Retriever(docs, embedder, config, namespace)
        print(namespace, r.dense.manifest())


if __name__ == "__main__":
    main()
