import json
import sqlite3

from nl2sql.config import DB_PATH, ROOT, db_checksum
from scripts.semantic_checks import check_semantics
from scripts.verify_data import overlap_report


def test_all_gold_semantics():
    sets = {
        n: json.loads((ROOT / f"data/{n}.json").read_text())
        for n in ("examples", "evaluation", "development")
    }
    with sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True) as c:
        checks = check_semantics(sets, c)
    assert len(checks) == 93 and all(r["passed"] for r in checks)


def test_checksum_and_labels():
    assert (
        db_checksum()
        == json.loads((ROOT / "data/provenance.json").read_text())["sha256"]
    )
    for r in json.loads((ROOT / "data/evaluation.json").read_text()):
        assert (
            r["references"]["supported"]
            and r["verification"]["human_review"] == "pending"
        )


def test_overlap_detector():
    q = {"id": "x", "question": "Count every track in the music catalog"}
    assert overlap_report([q], [q])
