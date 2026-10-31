"""Display helpers; no database or generation decisions live in the UI."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from nl2sql.config import ROOT


def show_result(result):
    st.caption(
        f"Status: {result.status} · API calls: {result.api_calls} · Cache: "
        + (", ".join(result.cache_status) or "not used")
    )
    if result.sql:
        st.code(result.sql, language="sql")
    if result.errors:
        for error in result.errors:
            st.warning(error)
    if result.model_diagnostics:
        with st.expander("Safe model failure diagnostics", expanded=True):
            st.json(result.model_diagnostics)
    if result.execution:
        execution = result.execution
        df = pd.DataFrame(execution.rows, columns=execution.columns)
        st.dataframe(df, alt="SQL query result rows")
        if execution.truncated:
            st.warning(
                "Result truncated to the display row limit. This is not a complete result."
            )
        st.write(result.summary)
        # Only a category plus a numeric measure with unique category labels.
        if len(df.columns) == 2 and 1 < len(df) <= 30 and df.columns.is_unique:
            x, y = df.columns
            if (
                not pd.api.types.is_numeric_dtype(df[x])
                and pd.api.types.is_numeric_dtype(df[y])
                and df[x].is_unique
            ):
                st.plotly_chart(
                    px.bar(df, x=x, y=y), alt="Returned measure by category"
                )
    with st.expander("Retrieval evidence, joins and retries"):
        for label, hits in [("Schema", result.schema), ("Examples", result.examples)]:
            st.markdown("**" + label + "**")
            if hits:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "id": h.document.id,
                                "stage": h.stage,
                                "score": h.score,
                                "text": h.document.text,
                            }
                            for h in hits
                        ]
                    ),
                    alt=label + " retrieval evidence",
                )
        st.json(
            {
                "foreign_key_expansion": result.expansion,
                "retrieval_stages": result.retrieval_trace,
                "retries": result.retries,
            }
        )
    with st.expander("Timing, tokens and configuration"):
        st.json(
            {
                "timings_seconds": result.timings,
                "token_usage": result.token_usage,
                "structured_summary": result.summary_details,
                "prompt_characters": result.prompt_chars,
                "configuration_fingerprint": result.fingerprint,
            }
        )
        st.caption(
            "Unavailable usage is null. Cache-hit timing is not a fresh-model latency measurement."
        )


def experiment_files():
    return sorted((ROOT / "results").glob("**/aggregate.json"))


def show_experiments():
    paths = experiment_files()
    if not paths:
        st.info(
            "No live experiment results saved yet. Run the experiment CLI after setting credentials."
        )
        return
    path = st.selectbox(
        "Saved experiment", paths, format_func=lambda p: str(p.relative_to(ROOT))
    )
    data = json.loads(path.read_text())
    records = []
    for name, agg in data["conditions"].items():
        acc = agg["execution_accuracy"] or {}
        records.append(
            {
                "condition": name,
                "n": agg["n"],
                "accuracy": acc.get("mean"),
                "CI low": acc.get("low"),
                "CI high": acc.get("high"),
                "cache hits": agg["cache_hits"],
            }
        )
    frame = pd.DataFrame(records)
    st.dataframe(frame, alt="Actual experiment aggregates")
    if frame["accuracy"].notna().any():
        st.plotly_chart(
            px.bar(frame, x="condition", y="accuracy"),
            alt="Measured execution accuracy by condition",
        )
    st.caption(
        "Exploratory comparisons on a small dataset. Point estimate differences alone do not establish significance."
    )
    st.json(data)


def show_error_analysis():
    paths = sorted((ROOT / "results").glob("**/*.jsonl"))
    if not paths:
        st.info(
            "No per-question experiment results available. Gold SQL appears only in this evaluation view."
        )
        return
    path = st.selectbox("Per-question results", paths)
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records:
        st.info("No completed questions.")
        return
    record = st.selectbox(
        "Question record",
        records,
        format_func=lambda r: r["condition"] + " / " + r["id"],
    )
    golds = {
        r["id"]: r
        for name in ("evaluation", "development")
        for r in json.loads((ROOT / f"data/{name}.json").read_text())
    }
    gold = golds.get(record["id"])
    st.write(record["result"]["question"])
    cols = st.columns(2)
    with cols[0]:
        st.caption("Generated SQL")
        st.code(record["result"]["sql"] or "No SQL", language="sql")
    with cols[1]:
        st.caption("Gold SQL · human review pending")
        if gold:
            st.code(gold["sql"], language="sql")
    st.caption("Automatic category: " + record["metrics"]["error_category"])
    st.selectbox(
        "Session-only reviewer category",
        [
            "unknown",
            "wrong table",
            "join",
            "aggregation",
            "filter",
            "syntax",
            "abstention",
            "resource limit",
        ],
        key="review_category",
    )
    st.json(
        {
            "metrics": record["metrics"],
            "errors": record["result"]["errors"],
            "retry_history": record["result"]["retries"],
        }
    )
