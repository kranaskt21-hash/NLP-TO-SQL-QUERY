from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'data/chinook.db'
BGE = 'BAAI/bge-small-en-v1.5'
MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
GEMINI_EMBED = 'gemini-embedding-2'
REVISIONS = {BGE: 'main', MINILM: 'main', GEMINI_EMBED: 'provider-managed'}
RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '
TABLES = {'Customer','Employee','Invoice','InvoiceLine','Track','Album','Artist','Genre','MediaType','Playlist','PlaylistTrack'}
@dataclass(frozen=True)
class Config:
    implementation: str = 'scratch'
    model: str = 'gemini-2.5-flash'
    embedding: str = MINILM
    retrieval: str = 'hybrid'
    schema_mode: str = 'retrieved'
    candidates: int = 20
    schema_k: int = 8
    examples_k: int = 4
    example_mode: str = 'retrieved'
    rrf_k: int = 60
    rerank: bool = False
    fk_expansion: bool = True
    corrections: int = 2
    distractors: int = 0
    seed: int = 42
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
        if self.distractors not in (0,25,50,100): raise ValueError('Invalid distractor count')
        if not 0 <= self.corrections <= 2: raise ValueError('At most two corrections')
        if self.retrieval not in ('dense','bm25','hybrid'): raise ValueError('Invalid retrieval')
        if self.implementation not in ('scratch','langchain'): raise ValueError('Invalid implementation')
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def db_checksum(): return hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
