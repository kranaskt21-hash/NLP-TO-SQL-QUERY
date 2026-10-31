"""Reproducible physical nonblank, non-comment Python line count (not complexity)."""

import json
from pathlib import Path


def count(path):
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def main():
    groups = {
        "scratch_and_shared_pipeline": list(Path("nl2sql").glob("*.py")),
        "langchain_adapters": list(Path("lc").glob("*.py")),
        "shared_evaluation_and_ui": list(Path("shared").glob("*.py")),
        "app": [Path("app.py")],
    }
    report = {name: {str(p): count(p) for p in paths} for name, paths in groups.items()}
    report["totals"] = {
        name: sum(count(p) for p in paths) for name, paths in groups.items()
    }
    Path("results/code_size.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
