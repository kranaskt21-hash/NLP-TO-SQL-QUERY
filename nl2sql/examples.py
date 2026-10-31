import json

from nl2sql.config import ROOT
from shared.interfaces import Document


def load_examples():
    # Evaluation and development paths are deliberately inaccessible here.
    rows = json.loads((ROOT / "data/examples.json").read_text(encoding="utf-8"))
    return [
        Document(r["id"], r["question"] + "\nSQL: " + r["sql"], kind="example")
        for r in rows
    ]
