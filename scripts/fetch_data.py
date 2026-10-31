import hashlib
import json
import sqlite3
import urllib.request
from pathlib import Path

root = Path("data")
commit = "7f67772503d71ba90f19283c38e93923addb43fa"
base = f"https://raw.githubusercontent.com/lerocha/chinook-database/{commit}/"
tree = json.load(
    urllib.request.urlopen(
        f"https://api.github.com/repos/lerocha/chinook-database/git/trees/{commit}?recursive=1"
    )
)
paths = [x["path"] for x in tree["tree"]]
print([p for p in paths if "Sqlite.sqlite" in p or "LICENSE" in p])
p = next(p for p in paths if p.endswith("Chinook_Sqlite.sqlite"))
raw = urllib.request.urlopen(base + p).read()
(root / "chinook.db").write_bytes(raw)
lic = next(p for p in paths if "LICENSE" in p.upper() and p.count("/") == 0)
(root / "CHINOOK_LICENSE.txt").write_bytes(urllib.request.urlopen(base + lic).read())
c = sqlite3.connect(root / "chinook.db")
print(c.execute("pragma integrity_check").fetchall())
print(c.execute("select name from sqlite_master where type='table'").fetchall())
(root / "provenance.json").write_text(
    json.dumps(
        dict(
            source=base + p,
            repository="https://github.com/lerocha/chinook-database",
            commit=commit,
            sha256=hashlib.sha256(raw).hexdigest(),
            license_source=base + lic,
            license="MIT",
            integrity_check=c.execute("pragma integrity_check").fetchone()[0],
        ),
        indent=2,
    )
)
