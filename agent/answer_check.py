"""Shared SQL execution and answer comparison logic."""
import sqlite3


def execute(conn: sqlite3.Connection, sql: str):
    """Execute SQL and return a scalar or sorted list of first-column values."""
    cur = conn.execute(sql)
    rows = cur.fetchall()
    if len(rows) == 1 and len(rows[0]) == 1:
        v = rows[0][0]
        return round(float(v), 2) if isinstance(v, float) else v
    return sorted((r[0] for r in rows), key=str)


def answers_match(actual, expected) -> bool:
    """Return True if actual matches expected within acceptable tolerance."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) < 0.01
    if isinstance(expected, list) and isinstance(actual, list):
        return sorted(str(x) for x in actual) == sorted(str(x) for x in expected)
    return str(actual) == str(expected)
