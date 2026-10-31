from dataclasses import replace

import pytest

from nl2sql.config import Config
from nl2sql.llm import Budget, CachedModel, Generation, ModelError


class Backend:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return Generation(status="sql", sql="SELECT 1"), {
            "input": 5,
            "output": 3,
            "total": 8,
        }


def test_cache_isolation(tmp_path, monkeypatch):
    import nl2sql.llm as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    backend = Backend()
    budget = Budget()
    config = Config()
    model = CachedModel(config, budget, backend)
    assert model.invoke("prompt", {"schema": "v1"})[2] == "fresh"
    assert model.invoke("prompt", {"schema": "v1"})[2] == "hit"
    assert backend.calls == budget.calls == 1
    assert model.invoke("prompt", {"schema": "v2"})[2] == "fresh"
    lc = CachedModel(replace(config, implementation="langchain"), budget, backend)
    assert lc.invoke("prompt", {"schema": "v1"})[2] == "fresh"
    assert model.invoke("changed prompt", {"schema": "v1"})[2] == "fresh"


def test_provider_error_not_leaked():
    class Bad:
        def invoke(self, prompt):
            raise RuntimeError("SECRET_MUST_NOT_APPEAR")

    with pytest.raises(ModelError) as exc:
        CachedModel(Config(cache=False), Budget(), Bad()).invoke("x", {})
    assert "SECRET_MUST_NOT_APPEAR" not in str(exc.value)


def test_auth_failure_not_retried_as_sql():
    class AuthError(Exception):
        code = 401

    class Unauthorized:
        def invoke(self, prompt):
            raise AuthError("do not expose response")

    from nl2sql.pipeline import Pipeline

    budget = Budget()
    result = Pipeline(
        Config(retrieval="bm25", cache=False), budget, Unauthorized()
    ).run("Count tracks")
    assert result.status == "model_error" and result.retries == [] and budget.calls == 1
