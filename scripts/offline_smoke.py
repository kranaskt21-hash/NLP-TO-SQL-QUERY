"""Real local dense retrieval plus MOCKED structured generation, not live accuracy."""

import json

from nl2sql.config import ROOT, Config
from nl2sql.llm import Generation
from nl2sql.pipeline import Pipeline


class MockGeneration:
    def invoke(self, prompt):
        return Generation(
            status="sql", sql="SELECT COUNT(*) AS Tracks FROM Track"
        ), None


def main():
    results = []
    for implementation in ("scratch", "langchain"):
        result = Pipeline(
            Config(implementation=implementation, cache=False), model=MockGeneration()
        ).run("How many tracks are in the catalog?")
        assert result.status == "success" and result.execution.rows == [(3503,)]
        results.append(
            {
                "implementation": implementation,
                "status": result.status,
                "schema_ids": [h.document.id for h in result.schema],
                "example_ids": [h.document.id for h in result.examples],
                "timings": result.timings,
                "generation": "mocked",
                "interpretation": "Sequential integration checks share model warmup; timing is not a pipeline performance comparison.",
                "token_usage": result.token_usage,
            }
        )
    assert (
        results[0]["schema_ids"] == results[1]["schema_ids"]
        and results[0]["example_ids"] == results[1]["example_ids"]
    )
    (ROOT / "results/offline_smoke.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
