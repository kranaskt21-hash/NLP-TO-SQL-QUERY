"""Explicit, budgeted connection check; no user question or SQL history sent."""

import re
from dataclasses import asdict, dataclass, field, replace

from nl2sql.llm import (
    BudgetError,
    CachedModel,
    MissingKey,
    ModelError,
    ScratchModel,
    get_api_key,
)
from nl2sql.model_errors import failure_message, safe_failure


@dataclass
class ConnectionCheck:
    status: str = "pending"
    stage: str = "list_models"
    model: str = ""
    implementation: str = ""
    available_models: list[str] = field(default_factory=list)
    api_calls: int = 0
    token_usage: dict | None = None
    message: str = ""
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def check_connection(config, budget):
    result = ConnectionCheck(model=config.model, implementation=config.implementation)
    initial = budget.calls
    try:
        from google import genai

        client = genai.Client(
            api_key=get_api_key(),
            vertexai=False,
            http_options={
                "retry_options": {"attempts": 1},
                "timeout": config.request_timeout_ms,
            },
        )
        try:
            budget.consume(0)
            pager = client.models.list(config={"page_size": 100})
            # Read only the fetched page. Iterating the pager could make unbudgeted calls.
            result.available_models = sorted(
                {
                    m.name.removeprefix("models/")
                    for m in pager.page
                    if m.name
                    and re.fullmatch(r"(?:models/)?gemini-[a-zA-Z0-9_.-]+", m.name)
                    and "generateContent" in (m.supported_actions or [])
                }
            )
        finally:
            client.close()
        result.stage = "structured_generation"
        c = replace(config, cache=False, transport_retries=0, corrections=0)
        backend = ScratchModel(c)
        if c.implementation == "langchain":
            from lc.chains import LangChainModel

            backend = LangChainModel(c)
        _, usage, _ = CachedModel(c, budget, backend).invoke(
            'Connection check. Return status="sql", sql="SELECT 1", reason="". No other content.',
            {"diagnostic": "connection-check-v1"},
        )
        result.token_usage = usage
        result.status = "success"
        result.message = "Gemini returned valid structured output through the selected implementation. Try your question again."
    except MissingKey as exc:
        result.status = "missing_key"
        result.message = str(exc)
    except BudgetError as exc:
        result.status = "budget"
        result.message = str(exc)
    except ModelError as exc:
        result.status = "error"
        result.message = str(exc)
        result.diagnostics = exc.diagnostics
    except Exception as exc:
        result.status = "error"
        result.diagnostics = safe_failure(exc)
        result.message = failure_message(result.diagnostics)
    result.api_calls = budget.calls - initial
    return result
