import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from nl2sql.config import REVISIONS, ROOT, Config, db_checksum, digest
from nl2sql.llm import Budget, BudgetError, MissingKey, ModelError, get_api_key
from nl2sql.pipeline import Pipeline
from shared.experiments import aggregate, metadata, variants
from shared.metrics import evaluate, paired_bootstrap


def main():
    p = argparse.ArgumentParser(
        description="Controlled ablations; defaults to a free dry run"
    )
    p.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    p.add_argument("--live", action="store_true")
    p.add_argument("--full-paid", action="store_true")
    p.add_argument("--call-budget", type=int)
    p.add_argument("--token-budget", type=int)
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "results/local/run")
    a = p.parse_args()
    full = a.suite == "full"
    if (
        full
        and a.live
        and (not a.full_paid or a.call_budget is None or a.token_budget is None)
    ):
        p.error("Full live mode needs --full-paid, --call-budget and --token-budget")
    configs = variants(full)
    if a.config:
        content = json.loads(a.config.read_text())
        configs = {name: Config(**cfg) for name, cfg in content.items()}
    questions = json.loads(
        (
            ROOT / ("data/evaluation.json" if full else "data/development.json")
        ).read_text()
    )
    estimate = sum(
        len(questions) * (1 + c.corrections) * (1 + c.transport_retries)
        for c in configs.values()
    )
    print(
        json.dumps(
            {
                "questions": len(questions),
                "conditions": len(configs),
                "maximum_generation_requests": estimate,
                "note": "Gemini embedding builds/query calls are additional and also budgeted. Cache misses cost calls. Token budget reserves UTF-8 bytes plus max output; actual usage recorded when available.",
            },
            indent=2,
        )
    )
    if not a.live:
        print("Dry run only. Add --live to execute.")
        return
    try:
        get_api_key()
    except MissingKey as exc:
        p.exit(2, str(exc) + "\n")
    budget = Budget(
        a.call_budget if a.call_budget is not None else 12,
        a.token_budget if a.token_budget is not None else 100000,
    )
    a.output.mkdir(parents=True, exist_ok=True)
    meta = metadata()
    meta["database_sha256"] = db_checksum()
    meta["embedding_revisions"] = REVISIONS
    meta["question_set_sha256"] = digest(questions)
    from nl2sql.prompt import RULES

    meta["instructions_sha256"] = digest(RULES)
    run_identity = {k: v for k, v in meta.items() if k != "timestamp"}
    meta["configs"] = {n: asdict(c) for n, c in configs.items()}
    (a.output / "configuration.json").write_text(json.dumps(meta, indent=2))
    all_records = []
    aggregates = {}
    preparation = {}
    for name, c in configs.items():
        path = a.output / (
            digest(
                {
                    "name": name,
                    "config": asdict(c),
                    "questions": questions,
                    "environment": run_identity,
                }
            )[:20]
            + ".jsonl"
        )
        records = (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )
        completed = {r["id"] for r in records}
        pipeline = Pipeline(c, budget)
        tick = time.perf_counter()
        before = budget.calls
        try:
            pipeline.prepare()
        except (BudgetError, ModelError, MissingKey) as exc:
            print("Preparation stopped:", type(exc).__name__)
            preparation[name] = {"status": type(exc).__name__}
            aggregates[name] = aggregate(records)
            all_records.extend(records)
            continue
        preparation[name] = {
            "status": "ready",
            "seconds": time.perf_counter() - tick,
            "api_calls": budget.calls - before,
        }
        for q in questions:
            if q["id"] in completed:
                continue
            result = pipeline.run(q["question"], evaluation=True)
            if result.status in ("budget", "missing_key", "model_error"):
                print("Stopped:", result.status, result.errors)
                break
            record = {
                "condition": name,
                "id": q["id"],
                "difficulty": q.get("difficulty", "development"),
                "result": result.to_dict(),
                "metrics": evaluate(result, q),
                "configuration": asdict(c),
                "run_metadata": meta,
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            records.append(record)
            print(name, q["id"], result.status)
        aggregates[name] = aggregate(records)
        all_records.extend(records)
    pairs = {}
    for name in configs:
        if name == "baseline":
            continue
        a_scores = {
            r["id"]: float(r["metrics"]["execution_accuracy"])
            for r in all_records
            if r["condition"] == name and r["metrics"]["execution_accuracy"] is not None
        }
        b_scores = {
            r["id"]: float(r["metrics"]["execution_accuracy"])
            for r in all_records
            if r["condition"] == "baseline"
            and r["metrics"]["execution_accuracy"] is not None
        }
        pairs[name + " minus baseline"] = paired_bootstrap(a_scores, b_scores)
    for noise in (0, 25, 50, 100):
        retrieved_name = f"schema=retrieved,noise={noise}"
        full_name = f"schema=full,noise={noise}"
        if retrieved_name not in configs or full_name not in configs:
            continue

        def condition_scores(name):
            return {
                r["id"]: float(r["metrics"]["execution_accuracy"])
                for r in all_records
                if r["condition"] == name
                and r["metrics"]["execution_accuracy"] is not None
            }

        pairs[f"retrieved minus full, noise={noise}"] = paired_bootstrap(
            condition_scores(retrieved_name), condition_scores(full_name)
        )
    out = {
        "metadata": meta,
        "conditions": aggregates,
        "paired_differences": pairs,
        "preparation": preparation,
        "budget": {"calls": budget.calls, "tokens_or_reservations": budget.tokens},
    }
    (a.output / "aggregate.json").write_text(json.dumps(out, indent=2))
    print("Saved", a.output)


if __name__ == "__main__":
    main()
