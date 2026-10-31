"""The executor always validates, including callers outside the UI."""

import sqlite3
import time
from contextlib import contextmanager

from nl2sql.config import DB_PATH, SQLITE_MAX_SQL_BYTES, SQLITE_MAX_VALUE_BYTES, TABLES
from shared.interfaces import Execution


class ResourceLimit(RuntimeError):
    pass


@contextmanager
def connection(deadline_s=3.0):
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    con.enable_load_extension(False)
    con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, SQLITE_MAX_VALUE_BYTES)
    con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, SQLITE_MAX_SQL_BYTES)
    con.execute("PRAGMA query_only=ON")
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    from nl2sql.validator import FUNCTIONS

    def authorize(action, arg1, arg2, database, trigger):
        if action == sqlite3.SQLITE_READ and arg1 not in TABLES:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() not in FUNCTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    con.set_authorizer(authorize)
    deadline = time.perf_counter() + deadline_s
    con.set_progress_handler(lambda: int(time.perf_counter() > deadline), 100)
    try:
        yield con
    finally:
        con.close()


def schema():
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {
            t: {r[1]: r[2] for r in con.execute(f'PRAGMA table_info("{t}")')}
            for t in tables
        }
    finally:
        con.close()


def execute(sql, *, row_limit=200, deadline_s=3.0, evaluation=False):
    from nl2sql.validator import validate

    validated = validate(sql)
    try:
        with connection(deadline_s) as con:
            cur = con.execute(validated)
            rows = cur.fetchmany(row_limit + 1)
            truncated = len(rows) > row_limit
            if truncated and evaluation:
                raise ResourceLimit(
                    "Evaluation result exceeds row budget; comparison unavailable"
                )
            return Execution(
                [c[0] for c in cur.description], rows[:row_limit], truncated
            )
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc):
            raise ResourceLimit("Query deadline exceeded") from None
        raise
