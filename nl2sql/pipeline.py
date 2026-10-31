import random
import sqlite3
import time

from nl2sql.config import Config, db_checksum, digest
from nl2sql.db import ResourceLimit, execute
from nl2sql.distractors import distractors
from nl2sql.embeddings import Embedder
from nl2sql.examples import load_examples
from nl2sql.fk_graph import expand
from nl2sql.llm import Budget, BudgetError, CachedModel, MissingKey, ModelError
from nl2sql.prompt import generation_prompt
from nl2sql.retrieval import Retriever
from nl2sql.schema_docs import documents
from nl2sql.validator import ValidationError, validate
from shared.interfaces import Hit, Result


class Pipeline:
    def __init__(self, config=Config(), budget=None, model=None):
        self.config = config
        self.budget = budget or Budget(config.session_calls, config.token_budget)
        self.injected_model = model

    def prepare(self):
        """Warm local resources/indexes outside evaluation question timing."""
        from nl2sql.config import GEMINI_EMBED

        c = self.config
        cls = Retriever
        if c.implementation == "langchain":
            from lc.retrievers import LangChainRetriever

            cls = LangChainRetriever
        embedder = Embedder(c.embedding, self.budget)
        needed = c.schema_mode == "retrieved" or (
            c.examples_k and c.example_mode == "retrieved"
        )
        if needed and c.retrieval != "bm25" and c.embedding != GEMINI_EMBED:
            embedder.encode(["Warmup schema retrieval"], query=True)
        if c.schema_mode == "retrieved":
            self.prepared_schema = cls(
                documents() + distractors(c.distractors, c.seed), embedder, c, "schema"
            )
        if c.examples_k and c.example_mode == "retrieved":
            self.prepared_examples = cls(load_examples(), embedder, c, "examples")
        if c.rerank and needed:
            from nl2sql.rerank import model

            model().predict([("Warmup", "Warmup document")])

    def run(self, question, *, evaluation=False):
        start = time.perf_counter()
        c = self.config
        r = Result(question, fingerprint=c.fingerprint())
        initial_calls = self.budget.calls
        try:
            if self.injected_model is None:
                from nl2sql.llm import get_api_key

                get_api_key()  # Fail before downloading models when no key exists.
            docs = documents() + distractors(c.distractors, c.seed)
            examples = load_examples()
            embedder = Embedder(c.embedding, self.budget)
            cls = Retriever
            if c.implementation == "langchain":
                from lc.retrievers import LangChainRetriever

                cls = LangChainRetriever
            if c.schema_mode == "full":
                r.schema = [Hit(d, 0.0, "full") for d in docs if d.kind == "table"]
            else:
                schema_retriever = getattr(self, "prepared_schema", None) or cls(
                    docs, embedder, c, "schema"
                )
                r.schema = schema_retriever.search(question, c.schema_k)
                r.retrieval_trace["schema"] = schema_retriever.trace
            if c.example_mode == "random":
                r.examples = [
                    Hit(d, 0.0, "random")
                    for d in random.Random(c.seed).sample(
                        examples, min(c.examples_k, len(examples))
                    )
                ]
            elif c.examples_k:
                retriever = getattr(self, "prepared_examples", None) or cls(
                    examples, embedder, c, "examples"
                )
                if c.implementation == "langchain":
                    from lc.example_selector import DynamicExampleSelector

                    chosen = DynamicExampleSelector(
                        retriever, c.examples_k
                    ).select_examples({"question": question})
                    mapping = {d.id: d for d in examples}
                    r.examples = [
                        Hit(mapping[x["id"]], x["score"], "dynamic") for x in chosen
                    ]
                else:
                    r.examples = retriever.search(question, c.examples_k)
                r.retrieval_trace["examples"] = retriever.trace
            r.timings["retrieval"] = time.perf_counter() - start
            noise_selected = [
                h.document.table for h in r.schema if h.document.id.startswith("noise:")
            ]
            if noise_selected:
                r.errors.append(
                    "Synthetic distractor schema selected (nonexistent tables): "
                    + ", ".join(noise_selected)
                )
            tick = time.perf_counter()
            if c.fk_expansion:
                r.expansion = expand([h.document.table for h in r.schema])
                tables = set(r.expansion["tables"])
                seen = {h.document.id for h in r.schema}
                # Complete table documents expose all columns for selected and bridge tables.
                r.schema.extend(
                    Hit(d, 0.0, "fk-completion")
                    for d in docs
                    if d.kind == "table" and d.table in tables and d.id not in seen
                )
            r.timings["fk_expansion"] = time.perf_counter() - tick
            backend = self.injected_model
            if backend is None and c.implementation == "langchain":
                from lc.chains import LangChainModel

                backend = LangChainModel(c)
            model = CachedModel(c, self.budget, backend)
            context = {
                "db": db_checksum(),
                "schema": digest([h.document.text for h in r.schema]),
                "examples": digest([h.document.text for h in r.examples]),
                "encoding": embedder.settings,
            }
            repair = None
            all_usage = []
            for attempt in range(c.corrections + 1):
                prompt = generation_prompt(question, r.schema, r.examples, repair)
                r.prompt_chars += len(prompt)
                tick = time.perf_counter()
                generation, usage, cache = model.invoke(prompt, context)
                r.timings["generation"] = (
                    r.timings.get("generation", 0) + time.perf_counter() - tick
                )
                r.cache_status.append(cache)
                all_usage.append(usage)
                if all(u is not None for u in all_usage):
                    r.token_usage = {
                        key: sum(u[key] for u in all_usage)
                        if all(u.get(key) is not None for u in all_usage)
                        else None
                        for key in ("input", "output", "total")
                    }
                else:
                    r.token_usage = None
                if generation.status == "abstain":
                    r.status = "abstain"
                    r.summary = generation.reason
                    break
                r.sql = generation.sql
                try:
                    tick = time.perf_counter()
                    validate(r.sql or "")
                    r.timings["validation"] = (
                        r.timings.get("validation", 0) + time.perf_counter() - tick
                    )
                    tick = time.perf_counter()
                    r.execution = execute(
                        r.sql,
                        row_limit=c.eval_rows if evaluation else c.ui_rows,
                        deadline_s=c.deadline_s,
                        evaluation=evaluation,
                    )
                    r.timings["execution"] = (
                        r.timings.get("execution", 0) + time.perf_counter() - tick
                    )
                    r.status = "success"
                    tick = time.perf_counter()
                    r.summary = grounded_summary(r.execution, c.summary_rows)
                    from nl2sql.summary import summary_payload

                    r.summary_details = summary_payload(
                        r.execution, c.summary_rows
                    ).model_dump()
                    r.timings["summary"] = time.perf_counter() - tick
                    break
                except ResourceLimit as exc:
                    r.status = "resource_limit"
                    r.errors.append(str(exc))
                    break
                except (ValidationError, sqlite3.Error) as exc:
                    error = str(exc)[:300]
                    r.errors.append(error)
                    r.retries.append(
                        {
                            "attempt": attempt,
                            "sql": r.sql,
                            "error": error,
                            "eligible": True,
                        }
                    )
                    if attempt == c.corrections:
                        r.status = "error"
                        break
                    repair = {
                        "previous_sql": r.sql,
                        "error": error,
                        "instruction": "Correct only this query within the supplied schema.",
                    }
        except MissingKey as exc:
            r.status = "missing_key"
            r.errors.append(str(exc))
        except BudgetError as exc:
            r.status = "budget"
            r.errors.append(str(exc))
        except ModelError as exc:
            r.model_diagnostics = exc.diagnostics
            r.status = "model_error"
            r.errors.append(str(exc))
        except Exception as exc:
            r.status = "error"
            r.errors.append("Pipeline failed: " + type(exc).__name__)
        r.api_calls = self.budget.calls - initial_calls
        r.timings["total"] = time.perf_counter() - start
        return r


def grounded_summary(execution, max_rows=20):
    # Deterministic result-grounded summary: no extra model call or unsupported claims.
    count = len(execution.rows)
    shown = execution.rows[:max_rows]
    if not count:
        return "The query returned no rows."
    import json

    scope = (
        f"{count} rows returned"
        + (" (display truncated)" if execution.truncated else "")
        + ". "
    )
    if len(shown) == 1:
        return scope + json.dumps(
            dict(zip(execution.columns, shown[0])), ensure_ascii=False, default=str
        )
    return (
        scope
        + f"Summary preview uses {len(shown)} rows"
        + (" only; preview is truncated." if count > max_rows else ".")
        + " First row: "
        + json.dumps(
            dict(zip(execution.columns, shown[0])), ensure_ascii=False, default=str
        )
    )
