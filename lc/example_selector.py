from langchain_core.example_selectors.base import BaseExampleSelector


class DynamicExampleSelector(BaseExampleSelector):
    def __init__(self, retriever, k):
        self.retriever = retriever
        self.k = k

    def add_example(self, example):
        raise NotImplementedError("Immutable training pool")

    def select_examples(self, input_variables):
        return [
            {"text": h.document.text, "id": h.document.id, "score": h.score}
            for h in self.retriever.search(input_variables["question"], self.k)
        ]
