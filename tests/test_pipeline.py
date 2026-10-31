import pytest

from nl2sql.config import Config
from nl2sql.llm import Budget, Generation, MissingKey, get_api_key
from nl2sql.pipeline import Pipeline
from shared.interfaces import PipelineProtocol, Result


class MockModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return next(self.outputs), None


@pytest.mark.parametrize("implementation", ["scratch", "langchain"])
def test_interface_and_retry(implementation):
    cfg = Config(implementation=implementation, retrieval="bm25", cache=False)
    model = MockModel(
        [
            Generation(status="sql", sql="SELECT Fake FROM Track"),
            Generation(status="sql", sql="SELECT COUNT(*) FROM Track"),
        ]
    )
    p: PipelineProtocol = Pipeline(cfg, model=model)
    r = p.run("Count catalog tracks")
    assert (
        isinstance(r, Result)
        and r.status == "success"
        and r.execution.rows == [(3503,)]
    )
    assert len(r.retries) == 1 and r.api_calls == 2 and r.token_usage is None
    assert len(model.prompts) == 2 and "previous_sql" in model.prompts[1]


def test_retry_bound():
    model = MockModel([Generation(status="sql", sql="SELECT Fake FROM Track")] * 4)
    r = Pipeline(Config(retrieval="bm25", cache=False), model=model).run("Question")
    assert r.status == "error" and len(model.prompts) == 3


def test_abstain_and_budget():
    model = MockModel([Generation(status="abstain", reason="No weather data")])
    r = Pipeline(Config(retrieval="bm25", cache=False), model=model).run("Weather?")
    assert r.status == "abstain" and r.execution is None
    b = Budget(calls=0)
    r = Pipeline(Config(retrieval="bm25", cache=False), budget=b, model=model).run(
        "Count tracks"
    )
    assert r.status == "budget"


def test_missing_key(monkeypatch):
    import streamlit as st

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(st, "secrets", {})
    with pytest.raises(MissingKey):
        get_api_key()
    r = Pipeline(Config(retrieval="bm25")).run("Count tracks")
    assert r.status == "missing_key" and r.api_calls == 0
    monkeypatch.setenv("GEMINI_API_KEY", "test-environment-key")
    assert get_api_key() == "test-environment-key"
    monkeypatch.setattr(st, "secrets", {"GEMINI_API_KEY": "test-secret-key"})
    assert get_api_key() == "test-secret-key"


def test_prompt_does_not_contain_gold():
    import json

    from nl2sql.config import ROOT

    model = MockModel([Generation(status="abstain", reason="test")])
    Pipeline(Config(retrieval="bm25", cache=False), model=model).run("Weather?")
    for row in json.loads((ROOT / "data/evaluation.json").read_text()):
        assert row["sql"] not in model.prompts[0]
