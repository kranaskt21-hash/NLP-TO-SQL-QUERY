from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class Document:
    id: str
    text: str
    table: str = ""
    kind: str = "table"


@dataclass
class Hit:
    document: Document
    score: float
    stage: str


@dataclass
class Execution:
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False


@dataclass
class Result:
    question: str
    status: str = "pending"
    sql: str | None = None
    execution: Execution | None = None
    retrieval_trace: dict = field(default_factory=dict)
    schema: list[Hit] = field(default_factory=list)
    examples: list[Hit] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    token_usage: dict[str, int | None] | None = None
    retries: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fingerprint: str = ""
    summary: str = ""
    summary_details: dict = field(default_factory=dict)
    expansion: dict = field(default_factory=dict)
    cache_status: list[str] = field(default_factory=list)
    prompt_chars: int = 0
    api_calls: int = 0

    def to_dict(self):
        return asdict(self)


class PipelineProtocol(Protocol):
    def run(self, question: str, *, evaluation: bool = False) -> Result: ...
