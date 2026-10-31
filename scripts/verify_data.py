"""Offline execution, reference labels, overlap audit, and independent semantic oracles."""

import json
import re
import sqlite3
from difflib import SequenceMatcher

from nl2sql.config import DB_PATH, ROOT, TABLES
from nl2sql.db import execute, schema
from nl2sql.validator import references


def overlap_report(examples, evaluation):
    pairs = []
    for a in examples:
        for b in evaluation:
            sa = set(re.findall(r"[a-z]+", a["question"].lower()))
            sb = set(re.findall(r"[a-z]+", b["question"].lower()))
            j = len(sa & sb) / len(sa | sb)
            ratio = SequenceMatcher(
                None, a["question"].lower(), b["question"].lower()
            ).ratio()
            if j > 0.55 or ratio > 0.72:
                pairs.append(
                    {
                        "retrieval": a["id"],
                        "evaluation": b["id"],
                        "jaccard": j,
                        "sequence_ratio": ratio,
                    }
                )
    return pairs


def main():
    assert set(schema()) == TABLES
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not con.execute("PRAGMA foreign_key_check").fetchall()
    groups = {
        n: json.loads((ROOT / f"data/{n}.json").read_text())
        for n in ("examples", "evaluation", "development")
    }
    checks = {}
    failures = []
    for name, rows in groups.items():
        for row in rows:
            try:
                result = execute(row["sql"], row_limit=100000, evaluation=True)
                row["references"] = references(row["sql"])
                row["verification"] = {
                    "automated_execution": "passed",
                    "semantic_checks": "see data/verification.json",
                    "human_review": "pending",
                }
                checks[row["id"]] = {
                    "execution": "passed",
                    "rows": len(result.rows),
                    "columns": len(result.columns),
                    "references_supported": row["references"]["supported"],
                }
            except Exception as exc:
                failures.append([row["id"], str(exc)])
        (ROOT / f"data/{name}.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    from scripts.semantic_checks import check_semantics

    semantic = check_semantics(groups, con)
    report = {
        "database_integrity": "passed",
        "foreign_keys": "passed",
        "execution": checks,
        "failures": failures,
        "semantic_checks": semantic,
        "human_review": "pending",
        "overlap_candidates": overlap_report(groups["examples"], groups["evaluation"]),
        "overlap_note": "Lexical heuristics flag candidates only; do not prove absence of semantic overlap.",
    }
    (ROOT / "data/verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "queries": len(checks),
                "failures": failures,
                "semantic": semantic,
                "overlap_candidates": len(report["overlap_candidates"]),
            },
            indent=2,
        )
    )
    if failures or any(not x["passed"] for x in semantic):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
