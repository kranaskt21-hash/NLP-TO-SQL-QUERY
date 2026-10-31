import json
from types import SimpleNamespace

import pytest

from nl2sql.config import Config
from nl2sql.diagnostics import check_connection
from nl2sql.llm import Budget, CachedModel, ModelError
from nl2sql.model_errors import safe_failure


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (400, "request_configuration"),
        (401, "authentication"),
        (403, "authentication"),
        (404, "model_unavailable"),
        (429, "quota"),
        (503, "service"),
    ],
)
def test_wrapped_status_without_provider_secrets(status, category):
    class Provider(Exception):
        code = status

    try:
        try:
            raise Provider("SECRET_MUST_NOT_APPEAR in request URL")
        except Provider as error:
            raise RuntimeError("SECRET_MUST_NOT_APPEAR") from error
    except RuntimeError as error:
        details = safe_failure(error)
    assert details["http_status"] == status and details["category"] == category
    assert "SECRET_MUST_NOT_APPEAR" not in json.dumps(details)
    assert details["exception_types"] == ["RuntimeError", "Provider"]


def test_runtime_error_keeps_cause_unknown():
    details = safe_failure(RuntimeError("SECRET_MUST_NOT_APPEAR"))
    assert details["category"] == "runtime" and details["http_status"] is None
    assert "SECRET_MUST_NOT_APPEAR" not in json.dumps(details)


def test_known_credentials_runtime():
    assert (
        safe_failure(RuntimeError("Could not resolve API token from the environment"))[
            "category"
        ]
        == "credentials_backend"
    )


def fake_client(monkeypatch, listing_error=None):
    from google import genai

    import nl2sql.llm as llm

    seen = []

    class Pager:
        page = [
            SimpleNamespace(
                name="models/gemini-test", supported_actions=["generateContent"]
            ),
            SimpleNamespace(
                name="models/gemini-embedding-001", supported_actions=["embedContent"]
            ),
        ]

        def __iter__(self):
            raise AssertionError("Do not fetch additional pages")

    def list_models(**kwargs):
        if listing_error is not None:
            raise listing_error
        seen.append("list")
        return Pager()

    def generate_content(**kwargs):
        seen.append("generate")
        return SimpleNamespace(
            text='{"status":"sql","sql":"SELECT 1","reason":""}', usage_metadata=None
        )

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["vertexai"] is False
            assert kwargs["http_options"]["retry_options"]["attempts"] == 1
            self.models = SimpleNamespace(
                list=list_models, generate_content=generate_content
            )

        def close(self):
            pass

    monkeypatch.setattr(genai, "Client", Client)
    monkeypatch.setattr(llm, "get_api_key", lambda: "mock-key")
    import nl2sql.diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "get_api_key", lambda: "mock-key")
    return seen


def test_connection_check_budget_and_models(monkeypatch):
    seen = fake_client(monkeypatch)
    budget = Budget(calls=2)
    result = check_connection(Config(), budget)
    assert result.status == "success" and result.api_calls == 2
    assert seen == ["list", "generate"] and result.available_models == ["gemini-test"]
    assert result.token_usage is None


def test_connection_check_fails_before_generation(monkeypatch):
    class Forbidden(Exception):
        code = 403

    seen = fake_client(monkeypatch, Forbidden("SECRET_MUST_NOT_APPEAR"))
    result = check_connection(Config(), Budget())
    assert (
        result.status == "error"
        and result.stage == "list_models"
        and result.api_calls == 1
    )
    assert result.diagnostics["category"] == "authentication" and seen == []
    assert "SECRET_MUST_NOT_APPEAR" not in json.dumps(result.to_dict())


def test_connection_check_quota_stops_second_call(monkeypatch):
    seen = fake_client(monkeypatch)
    result = check_connection(Config(), Budget(calls=1))
    assert result.status == "budget" and result.api_calls == 1 and seen == ["list"]


def test_wrapped_rate_limit_obeys_shared_budget(monkeypatch):
    import nl2sql.llm as module

    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    class Quota(Exception):
        code = 429

    class Backend:
        def invoke(self, prompt):
            try:
                raise Quota("SECRET_MUST_NOT_APPEAR")
            except Quota as error:
                raise RuntimeError("wrapper") from error

    budget = Budget()
    with pytest.raises(ModelError) as error:
        CachedModel(Config(cache=False), budget, Backend()).invoke("question", {})
    assert budget.calls == 2 and error.value.diagnostics["http_status"] == 429
    assert "SECRET_MUST_NOT_APPEAR" not in str(error.value)


def test_real_sdk_keeps_transport_open_until_request_finishes(monkeypatch):
    import httpx
    from google import genai

    import nl2sql.llm as llm

    real_client = genai.Client
    requests = []

    def respond(request):
        requests.append(request.url.host)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": '{"status":"sql","sql":"SELECT 1","reason":""}'
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            },
        )

    class Transport(httpx.MockTransport):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    transport = Transport(respond)

    def client_factory(**kwargs):
        options = kwargs["http_options"] | {"client_args": {"transport": transport}}
        # Do not retain the returned client here: that would mask premature disposal.
        return real_client(**(kwargs | {"http_options": options}))

    monkeypatch.setattr(genai, "Client", client_factory)
    monkeypatch.setattr(llm, "get_api_key", lambda: "offline-placeholder")
    generation, usage = llm.ScratchModel(Config()).invoke("Return SELECT 1")
    assert generation.sql == "SELECT 1" and usage["total"] == 15
    assert requests == ["generativelanguage.googleapis.com"] and transport.closed
