from types import SimpleNamespace

from nl2sql.config import GEMINI_EMBED, Config
from nl2sql.llm import Budget, Generation


def test_actual_langchain_structured_binding_without_network():
    from langchain_core.runnables import Runnable
    from langchain_google_genai import ChatGoogleGenerativeAI

    c = Config()
    model = ChatGoogleGenerativeAI(
        model=c.model,
        api_key="offline-placeholder",
        temperature=c.temperature,
        max_tokens=c.max_output_tokens,
        max_retries=0,
        thinking_level=c.thinking_level,
    )
    assert isinstance(
        model.with_structured_output(
            Generation, method="json_schema", include_raw=True
        ),
        Runnable,
    )


def test_scratch_structured_adapter(monkeypatch):
    from google import genai

    import nl2sql.llm as module

    seen = {}

    def generate_content(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            text='{"status":"sql","sql":"SELECT 1","reason":""}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5, total_token_count=15
            ),
        )

    class Client:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=generate_content)

    Client.close = lambda self: None
    monkeypatch.setattr(genai, "Client", Client)
    monkeypatch.setattr(module, "get_api_key", lambda: "mock-key")
    generation, usage = module.ScratchModel(Config()).invoke("question")
    assert generation.sql == "SELECT 1" and usage["total"] == 15
    assert (
        seen["config"].temperature == 0
        and seen["config"].response_mime_type == "application/json"
    )


def test_lcel_structured_adapter(monkeypatch):
    import langchain_google_genai
    from langchain_core.runnables import RunnableLambda

    import lc.chains as module

    seen = []

    class Fake:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema, **kwargs):
            def output(prompt):
                seen.append(prompt.to_messages()[0].content)
                return {
                    "parsed": Generation(status="sql", sql="SELECT 1"),
                    "raw": SimpleNamespace(
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        }
                    ),
                }

            return RunnableLambda(output)

    monkeypatch.setattr(module, "get_api_key", lambda: "mock-key")
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", Fake)
    generation, usage = module.LangChainModel(Config()).invoke("exact question data")
    assert (
        seen == ["exact question data"]
        and generation.sql == "SELECT 1"
        and usage["total"] == 15
    )


def test_remote_embedding_budget_and_tasks(monkeypatch):
    from google import genai

    import nl2sql.llm as module
    from nl2sql.embeddings import Embedder

    seen = []

    def embed_content(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1.0, 2.0]) for _ in kwargs["contents"]]
        )

    class Client:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(embed_content=embed_content)

    Client.close = lambda self: None
    monkeypatch.setattr(genai, "Client", Client)
    monkeypatch.setattr(module, "get_api_key", lambda: "mock-key")
    budget = Budget(calls=5)
    embedder = Embedder(GEMINI_EMBED, budget)
    assert embedder.encode(["doc"] * 65).shape == (65, 2)
    embedder.encode(["question"], query=True)
    assert budget.calls == 4
    assert (
        seen[0]["config"]["task_type"] == "RETRIEVAL_DOCUMENT"
        and seen[-1]["config"]["task_type"] == "RETRIEVAL_QUERY"
    )
