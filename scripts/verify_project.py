"""Repeatable offline verification report; external generation stays mocked."""

import json
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from nl2sql.config import ROOT, db_checksum
from nl2sql.llm import MissingKey, get_api_key


def main():
    (ROOT / ".cache").mkdir(exist_ok=True)
    commands = [
        (
            "syntax",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "app.py",
                "nl2sql",
                "lc",
                "shared",
                "scripts",
                "tests",
            ],
        ),
        ("imports", [sys.executable, "-m", "scripts.check_imports"]),
        ("gold_verification", [sys.executable, "-m", "scripts.verify_data"]),
        (
            "tests",
            [sys.executable, "-m", "pytest", "-q", "--junitxml=.cache/pytest.xml"],
        ),
        ("smoke_dry_run", [sys.executable, "-m", "scripts.run_experiments"]),
        (
            "full_dry_run",
            [sys.executable, "-m", "scripts.run_experiments", "--suite", "full"],
        ),
    ]
    if shutil.which("uv"):
        commands.append(
            (
                "dependency_compatibility",
                ["uv", "pip", "check", "--python", sys.executable],
            )
        )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "database_sha256": db_checksum(),
        "checks": {},
        "generation_test_mode": "mocked; no live accuracy measurement",
    }
    for name, cmd in commands:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        report["checks"][name] = {
            "passed": result.returncode == 0,
            "exit_code": result.returncode,
        }
        print(name, "PASS" if result.returncode == 0 else "FAIL")
        if result.returncode:
            print(result.stdout[-1500:])
            print(result.stderr[-1500:])
    if (ROOT / ".cache/pytest.xml").exists():
        suites = ET.parse(ROOT / ".cache/pytest.xml").getroot()
        counts = {
            key: sum(int(s.get(key, 0)) for s in suites.iter("testsuite"))
            for key in ("tests", "failures", "errors", "skipped")
        }
        report["pytest"] = counts
    try:
        get_api_key()
        report["live_api"] = "not run; key exists, use explicit smoke command"
    except MissingKey:
        report["live_api"] = "skipped: no usable GEMINI_API_KEY"
    report["human_gold_review"] = "pending"
    report["deployment"] = "not performed; no authenticated Community Cloud access"
    report["warning"] = (
        "langchain-community emits a sunset deprecation warning; pinned imports and components pass offline checks"
    )
    (ROOT / "results/verification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not all(check["passed"] for check in report["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
