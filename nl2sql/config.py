import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data/chinook.db"
BGE = "BAAI/bge-small-en-v1.5"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_EMBED = "gemini-embedding-001"
REVISIONS = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "gemini-embedding-001": "provider-managed",
}
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TABLES = {
    "Customer",
    "Employee",
    "Invoice",
    "InvoiceLine",
    "Track",
    "Album",
    "Artist",
    "Genre",
    "MediaType",
    "Playlist",
    "PlaylistTrack",
}
SQLITE_MAX_VALUE_BYTES = 1_000_000
SQLITE_MAX_SQL_BYTES = 100_000
EMBEDDING_BATCH_SIZE = 32


@dataclass(frozen=True)
class Config:
    implementation: str = "scratch"
    model: str = "gemini-3.8-flash"
    thinking_level: str = "low"
    embedding: str = MINILM
    retrieval: str = "hybrid"
    schema_mode: str = "retrieved"
    candidates: int = 20
    schema_k: int = 8
    examples_k: int = 4
    example_mode: str = "retrieved"
    rrf_k: int = 60
    rerank: bool = False
    fk_expansion: bool = True
    corrections: int = 2
    distractors: int = 0
    seed: int = 42
    request_timeout_ms: int = 30000
    temperature: float = 0.0
    max_output_tokens: int = 2048
    deadline_s: float = 3.0
    ui_rows: int = 200
    eval_rows: int = 100000
    summary_rows: int = 20
    session_calls: int = 20
    token_budget: int = 100000
    transport_retries: int = 1
    cache: bool = True

    def __post_init__(self):
        if self.distractors not in (0, 25, 50, 100):
            raise ValueError("Invalid distractor count")
        if not 0 <= self.corrections <= 2:
            raise ValueError("At most two corrections")
        if self.retrieval not in ("dense", "bm25", "hybrid"):
            raise ValueError("Invalid retrieval")
        if self.implementation not in ("scratch", "langchain"):
            raise ValueError("Invalid implementation")
        if self.schema_mode not in ("full", "retrieved") or self.example_mode not in (
            "random",
            "retrieved",
        ):
            raise ValueError("Invalid context selection mode")
        if (
            not 0 <= self.examples_k <= 30
            or min(self.candidates, self.schema_k, self.ui_rows, self.eval_rows) < 1
        ):
            raise ValueError("Invalid retrieval or row budget")
        if self.deadline_s <= 0 or self.transport_retries < 0:
            raise ValueError("Invalid timeout or retry budget")

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def db_checksum():
    return hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
