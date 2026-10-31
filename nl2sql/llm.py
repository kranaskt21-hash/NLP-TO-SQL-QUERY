import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from nl2sql.config import ROOT, digest


class MissingKey(ValueError):
    pass


class BudgetError(RuntimeError):
    pass


class ModelError(RuntimeError):
    pass


class Generation(BaseModel):
    status: Literal["sql", "abstain"]
    sql: str | None = None
    reason: str = ""


def get_api_key():
    key = None
    try:
        import streamlit as st

        key = st.secrets.get("GEMINI_API_KEY")
    except (FileNotFoundError, KeyError):
        pass
    if not key or key.startswith("REPLACE_"):
        key = os.environ.get("GEMINI_API_KEY")
    if not key or key.startswith("REPLACE_"):
        raise MissingKey("Set GEMINI_API_KEY in Streamlit secrets or the environment.")
    return key


class Budget:
    def __init__(self, calls=20, tokens=100000):
        self.limit = calls
        self.token_limit = tokens
        self.calls = 0
        self.tokens = 0
        self.lock = threading.Lock()

    def consume(self, reserved_tokens):
        with self.lock:
            if (
                self.calls >= self.limit
                or self.tokens + reserved_tokens > self.token_limit
            ):
                raise BudgetError("API call or reserved-token budget exhausted")
            self.calls += 1
            self.tokens += reserved_tokens

    def settle(self, reserved, actual):
        # Keep conservative reservation if provider usage is unavailable.
        if actual is not None:
            with self.lock:
                self.tokens += actual - reserved


class ScratchModel:
    def __init__(self, config):
        self.config = config

    def invoke(self, prompt):
        from google import genai
        from google.genai import types

        response = genai.Client(
            api_key=get_api_key(), http_options={"retry_options": {"attempts": 1}}
        ).models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                response_mime_type="application/json",
                response_schema=Generation,
                thinking_config=types.ThinkingConfig(
                    thinking_level=self.config.thinking_level
                ),
            ),
        )
        usage = response.usage_metadata
        return Generation.model_validate_json(response.text), (
            {
                "input": usage.prompt_token_count,
                "output": usage.candidates_token_count,
                "total": usage.total_token_count,
            }
            if usage
            else None
        )


class CachedModel:
    def __init__(self, config, budget, backend=None):
        self.config = config
        self.budget = budget
        self.backend = backend or ScratchModel(config)

    def invoke(self, prompt, context):
        c = self.config
        key = digest(
            {
                "prompt": prompt,
                "config": c.fingerprint(),
                "context": context,
                "format": 2,
            }
        )
        path = ROOT / ".cache/responses" / f"{key}.json"
        if c.cache and path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return Generation.model_validate(obj["generation"]), obj["usage"], "hit"
        # UTF-8 byte length is a conservative input-token reservation, not measured usage.
        reserved = len(prompt.encode("utf-8")) + c.max_output_tokens
        for attempt in range(c.transport_retries + 1):
            self.budget.consume(reserved)
            try:
                generation, usage = self.backend.invoke(prompt)
                self.budget.settle(reserved, usage.get("total") if usage else None)
                if c.cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=path.parent,
                        suffix=".tmp",
                        delete=False,
                    ) as handle:
                        json.dump(
                            {"generation": generation.model_dump(), "usage": usage},
                            handle,
                        )
                        temporary = Path(handle.name)
                    temporary.replace(path)
                return generation, usage, "fresh"
            except (MissingKey, BudgetError):
                raise
            except Exception as exc:
                # Provider messages may echo inputs or credentials: never persist them.
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if str(status) in ("429", "503") and attempt < c.transport_retries:
                    time.sleep(min(2**attempt, 4))
                    continue
                raise ModelError(
                    "Model request failed ("
                    + str(status or type(exc).__name__)
                    + "). Check key, quota, model availability or connection."
                ) from None
