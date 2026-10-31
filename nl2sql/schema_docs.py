import json
import sqlite3

from nl2sql.config import DB_PATH
from shared.interfaces import Document

DESCRIPTIONS = {
    "Customer": "Music store customers and assigned sales support representative",
    "Employee": "Store employees, reporting hierarchy and support agents",
    "Invoice": "Customer purchase headers, billing address, date and total",
    "InvoiceLine": "Purchased tracks with sale unit price and quantity",
    "Track": "Music tracks, duration in milliseconds, size in bytes and catalog unit price",
    "Album": "Albums linked to recording artists",
    "Artist": "Recording artists",
    "Genre": "Music genre classification",
    "MediaType": "Audio/video encoding formats",
    "Playlist": "Named music playlists",
    "PlaylistTrack": "Many-to-many playlist track membership",
}


def documents():
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    docs = []
    try:
        for table, description in DESCRIPTIONS.items():
            cols = list(con.execute(f'PRAGMA table_info("{table}")'))
            fks = list(con.execute(f'PRAGMA foreign_key_list("{table}")'))
            relations = [f"{table}.{f[3]} -> {f[2]}.{f[4]}" for f in fks]
            text = (
                f"Table {table}: {description}. Columns: "
                + ", ".join(
                    f"{c[1]} {c[2]}" + (" PRIMARY KEY" if c[5] else "") for c in cols
                )
                + ". Foreign keys: "
                + ", ".join(relations)
            )
            docs.append(Document("table:" + table, text, table))
            for c in cols:
                vals = [
                    r[0]
                    for r in con.execute(
                        f'SELECT DISTINCT "{c[1]}" FROM "{table}" WHERE "{c[1]}" IS NOT NULL ORDER BY "{c[1]}" LIMIT 3'
                    )
                ]
                docs.append(
                    Document(
                        f"column:{table}.{c[1]}",
                        f"{table}.{c[1]} {c[2]}; {description}; primary_key={bool(c[5])}; foreign keys={relations}; sample values={json.dumps(vals, ensure_ascii=False)[:180]}",
                        table,
                        "column",
                    )
                )
        return docs
    finally:
        con.close()
