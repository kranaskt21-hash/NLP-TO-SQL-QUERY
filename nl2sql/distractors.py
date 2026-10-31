import random

from shared.interfaces import Document


def distractors(count, seed=42):
    if count not in (0, 25, 50, 100):
        raise ValueError("Use 0,25,50,100")
    rng = random.Random(seed)
    sectors = [
        "Warehouse",
        "Campaign",
        "Shipment",
        "Subscription",
        "Supplier",
        "Insurance",
        "Telemetry",
        "Forecast",
    ]
    return [
        Document(
            f"noise:{i}",
            f"Table {name}: synthetic unrelated business schema. Columns: {name}Id INTEGER PRIMARY KEY, Region TEXT, Status TEXT, CreatedAt DATETIME, Amount NUMERIC.",
            name,
        )
        for i in range(count)
        for name in [rng.choice(sectors) + f"Archive{i:03}"]
    ]
