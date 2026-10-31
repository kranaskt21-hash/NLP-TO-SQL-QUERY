import importlib.metadata
import platform
from dataclasses import replace
from datetime import datetime, timezone

from nl2sql.config import BGE, GEMINI_EMBED, Config
from shared.metrics import bootstrap


def variants(full=False):
    base = Config()
    configs = {"baseline": base, "langchain": replace(base, implementation="langchain")}
    factors = {
        "retrieval": ["dense", "bm25"],
        "rerank": [True],
        "examples_k": [0, 2, 8],
        "example_mode": ["random"],
        "fk_expansion": [False],
        "corrections": [0, 1],
        "embedding": [BGE, GEMINI_EMBED],
    }
    if full:
        for field, values in factors.items():
            for value in values:
                configs[f"{field}={value}"] = replace(base, **{field: value})
        # Matched full/retrieved pairs at each noise level.
        for noise in (0, 25, 50, 100):
            for mode in ("full", "retrieved"):
                configs[f"schema={mode},noise={noise}"] = replace(
                    base, distractors=noise, schema_mode=mode
                )
    return configs


def aggregate(records):
    metrics = [
        "execution_accuracy",
        "sql_exact_match",
        "table_recall",
        "column_recall",
        "parse_valid",
        "execution_success",
        "abstention",
    ]
    out = {"n": len(records)}
    for metric in metrics:
        vals = [
            r["metrics"][metric] for r in records if r["metrics"][metric] is not None
        ]
        out[metric] = bootstrap(vals)
    eligible = [r for r in records if r["metrics"]["eligible_initial_failure"]]
    out["self_correction_success"] = bootstrap(
        [r["metrics"]["correction_success"] for r in eligible]
    )
    out["resource_limits"] = sum(r["metrics"]["resource_limit"] for r in records)
    out["distractor_selection_errors"] = sum(
        r["metrics"].get("distractor_selection_error", False) for r in records
    )
    out["difficulty"] = {
        d: bootstrap(
            [
                r["metrics"]["execution_accuracy"]
                for r in records
                if r["difficulty"] == d
                and r["metrics"]["execution_accuracy"] is not None
            ]
        )
        for d in ("easy", "medium", "hard", "development")
    }
    # Fresh-only latency: never mix cache-hit and fresh latency.
    out["latency_fresh_seconds"] = bootstrap(
        [
            r["result"]["timings"]["total"]
            for r in records
            if r["result"]["cache_status"]
            and all(s == "fresh" for s in r["result"]["cache_status"])
        ]
    )
    out["cache_hits"] = sum("hit" in r["result"]["cache_status"] for r in records)
    out["tokens_available"] = sum(
        r["result"]["token_usage"] is not None
        and r["result"]["token_usage"].get("total") is not None
        for r in records
    )
    out["tokens_total"] = (
        sum(
            r["result"]["token_usage"]["total"]
            for r in records
            if r["result"]["token_usage"] is not None
            and r["result"]["token_usage"].get("total") is not None
        )
        if out["tokens_available"]
        else None
    )
    out["token_provenance"] = (
        "Successful generation usage; cached entries retain original usage. Excludes unreported embedding/failed-request usage."
    )
    fresh = [
        r
        for r in records
        if r["result"]["cache_status"]
        and all(s == "fresh" for s in r["result"]["cache_status"])
    ]
    out["stage_latency_fresh_seconds"] = {
        stage: bootstrap(
            [
                r["result"]["timings"][stage]
                for r in fresh
                if stage in r["result"]["timings"]
            ]
        )
        for stage in (
            "retrieval",
            "fk_expansion",
            "generation",
            "validation",
            "execution",
            "summary",
            "total",
        )
    }
    out["cost_usd"] = None
    out["cost_reason"] = "No configured sourced pricing table"
    return out


def metadata():
    packages = [
        "streamlit",
        "numpy",
        "pandas",
        "sqlglot",
        "faiss-cpu",
        "google-genai",
        "sentence-transformers",
        "langchain-core",
        "langchain-community",
        "langchain-google-genai",
    ]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "software": {p: importlib.metadata.version(p) for p in packages},
        "seed": 42,
        "interpretation": "Exploratory; 60 questions, multiple comparisons, human gold review pending",
    }
