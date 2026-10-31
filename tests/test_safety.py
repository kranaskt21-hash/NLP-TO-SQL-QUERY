import sqlite3

import pytest

from nl2sql.config import DB_PATH, TABLES
from nl2sql.db import ResourceLimit, connection, execute, schema
from nl2sql.validator import ValidationError, references, validate


def test_integrity():
    with sqlite3.connect(DB_PATH) as c:
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not c.execute("PRAGMA foreign_key_check").fetchall()
    assert set(schema()) == TABLES


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE Track",
        "DELETE FROM Track",
        "PRAGMA table_info(Track)",
        "ATTACH DATABASE ':memory:' AS x",
        "SELECT 1; SELECT 2",
        "SELECT load_extension('x')",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM Fake",
        "SELECT Unknown FROM Track",
        "SELECT t.Unknown FROM Track t",
        "SELECT * FROM main.Track",
        "WITH x AS (DELETE FROM Track RETURNING *) SELECT * FROM x",
        "SELECT writefile('x','y')",
    ],
)
def test_invalid(sql):
    with pytest.raises(ValidationError):
        validate(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT c.CustomerId FROM Customer AS c",
        "WITH x AS (SELECT TrackId FROM Track) SELECT TrackId FROM x",
        'SELECT "TrackId" FROM "Track"',
        "SELECT TrackId FROM Track UNION SELECT TrackId FROM Track",
        "SELECT g.GenreId FROM Genre g WHERE NOT EXISTS (SELECT 1 FROM Track t WHERE t.GenreId=g.GenreId)",
        "SELECT COUNT(*) FROM Track",
    ],
)
def test_read_queries(sql):
    assert execute(sql).columns


def test_authorizer_and_readonly():
    with connection() as c:
        for sql in [
            "DELETE FROM Track",
            "CREATE TABLE x(a)",
            "ATTACH DATABASE ':memory:' AS x",
            "PRAGMA user_version=2",
            "SELECT load_extension('x')",
        ]:
            with pytest.raises(sqlite3.DatabaseError):
                c.execute(sql)
    with sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True) as c:
        with pytest.raises(sqlite3.OperationalError):
            c.execute("DELETE FROM Track")


def test_row_limits():
    result = execute("SELECT TrackId FROM Track", row_limit=3)
    assert result.truncated and len(result.rows) == 3
    with pytest.raises(ResourceLimit):
        execute("SELECT TrackId FROM Track", row_limit=3, evaluation=True)


def test_deadline():
    with pytest.raises(ResourceLimit):
        execute(
            "SELECT COUNT(*) FROM Track a CROSS JOIN Track b CROSS JOIN Track c",
            deadline_s=0.001,
        )


def test_references():
    r = references(
        "WITH x AS (SELECT TrackId,AlbumId FROM Track) SELECT a.Title,x.TrackId FROM x JOIN Album a ON x.AlbumId=a.AlbumId"
    )
    assert r["supported"] and set(r["tables"]) == {"Track", "Album"}
    assert {"Track.TrackId", "Track.AlbumId", "Album.Title", "Album.AlbumId"} <= set(
        r["columns"]
    )
    r = references("SELECT * FROM Genre")
    assert r["columns"] == ["Genre.GenreId", "Genre.Name"]
