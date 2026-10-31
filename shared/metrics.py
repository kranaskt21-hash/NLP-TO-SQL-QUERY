"""Execution comparisons preserve shape, column position, NULL and multiplicity."""

import math
import numbers
from collections import Counter

import numpy as np
import sqlglot

from nl2sql.validator import references
from shared.interfaces import Execution


def cell_equal(a, b, rtol=1e-6, atol=1e-8):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return type(a) is type(b) and a == b


def execution_equal(pred, gold, ordered=False, tie_groups=None):
    if pred.truncated or gold.truncated:
        return None
    if len(pred.columns) != len(gold.columns) or len(pred.rows) != len(gold.rows):
        return False

    def eq(a, b):
        return len(a) == len(b) and all(cell_equal(x, y) for x, y in zip(a, b))

    if ordered:
        if tie_groups is None:
            return all(eq(a, b) for a, b in zip(pred.rows, gold.rows))
        covered = 0
        for start, end in tie_groups:
            if start != covered or end <= start or end > len(gold.rows):
                raise ValueError("Tie groups must partition all result rows")
            covered = end
        if covered != len(gold.rows):
            raise ValueError("Tie groups must partition all result rows")
        # Explicit gold order-key groups may permit permutations within ties only.
        return all(
            execution_equal(
                Execution(pred.columns, pred.rows[a:b]),
                Execution(gold.columns, gold.rows[a:b]),
            )
            for a, b in tie_groups
        )
    if Counter(map(tuple, pred.rows)) == Counter(map(tuple, gold.rows)):
        return True
    # Bipartite matching avoids greedy failures under floating point tolerances.
    adjacency = [[j for j, b in enumerate(gold.rows) if eq(a, b)] for a in pred.rows]
    match = {}

    def augment(i, seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match or augment(match[j], seen):
                match[j] = i
                return True
        return False

    return all(augment(i, set()) for i in range(len(pred.rows)))


def normalized_sql(sql):
    try:
        return sqlglot.parse_one(sql, read="sqlite").sql(
            dialect="sqlite", normalize=True, pretty=False
        )
    except Exception:
        return None


def bootstrap(values, seed=42, n=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(n, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "n": len(values),
    }


def paired_bootstrap(a, b, seed=42):
    common = sorted(set(a) & set(b))
    return bootstrap([a[k] - b[k] for k in common], seed) if common else None


def evaluate(result, gold):
    from nl2sql.db import execute

    gold_exec = execute(gold["sql"], row_limit=100000, evaluation=True)
    tree = sqlglot.parse_one(gold["sql"], read="sqlite")
    ordered = tree.args.get("order") is not None
    correct = (
        execution_equal(result.execution, gold_exec, ordered)
        if result.execution
        else (None if result.status == "resource_limit" else False)
    )
    pred_refs = references(result.sql) if result.sql else {"supported": False}
    expected = gold.get("references") or references(gold["sql"])
    from nl2sql.db import schema

    catalog = schema()
    selected_tables = {h.document.table for h in result.schema}
    selected_cols = set()
    for h in result.schema:
        d = h.document
        if d.kind == "table" and d.table in catalog:
            selected_cols.update(d.table + "." + c for c in catalog[d.table])
        elif d.kind == "column":
            selected_cols.add(d.id.removeprefix("column:"))

    def recall(expected, selected):
        return len(set(expected) & selected) / len(set(expected)) if expected else None

    return {
        "execution_accuracy": correct,
        "sql_exact_match": normalized_sql(result.sql) == normalized_sql(gold["sql"])
        if result.sql
        else False,
        "parse_valid": parse_valid(result.sql),
        "execution_success": result.status == "success",
        "table_recall": recall(expected["tables"], selected_tables)
        if expected["supported"]
        else None,
        "column_recall": recall(expected["columns"], selected_cols)
        if expected["supported"]
        else None,
        "reference_extraction_supported": pred_refs["supported"],
        "eligible_initial_failure": bool(result.retries),
        "correction_success": bool(result.retries) and result.status == "success",
        "abstention": result.status == "abstain",
        "resource_limit": result.status == "resource_limit",
        "distractor_selection_error": any(
            h.document.id.startswith("noise:") for h in result.schema
        ),
        "error_category": categorize(result, pred_refs, expected),
    }


def parse_valid(sql):
    try:
        return sql is not None and sqlglot.parse_one(sql, read="sqlite") is not None
    except Exception:
        return False


def categorize(result, pred, expected):
    if result.status == "abstain":
        return "abstention (heuristic)"
    if result.status == "resource_limit":
        return "resource limit (heuristic)"
    if result.sql and not parse_valid(result.sql):
        return "syntax (heuristic)"
    if (
        pred.get("supported")
        and expected.get("supported")
        and set(pred["tables"]) != set(expected["tables"])
    ):
        return "wrong table (heuristic)"
    return "unknown"
