"""SQLite storage. One file, no ORM: the schema is small and the queries are simple."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS shops (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    trade TEXT NOT NULL DEFAULT 'plumbing',
    owner_name TEXT NOT NULL DEFAULT '',
    owner_phone TEXT NOT NULL,
    twilio_number TEXT UNIQUE,
    timezone TEXT NOT NULL DEFAULT 'America/Chicago',
    service_area TEXT NOT NULL DEFAULT '',
    services TEXT NOT NULL DEFAULT '',
    pricing_notes TEXT NOT NULL DEFAULT '',
    extra_instructions TEXT NOT NULL DEFAULT '',
    emergency_keywords TEXT NOT NULL DEFAULT '',
    business_hours TEXT NOT NULL DEFAULT '{}',
    slot_minutes INTEGER NOT NULL DEFAULT 120,
    calendar_provider TEXT NOT NULL DEFAULT 'local',
    google_calendar_id TEXT NOT NULL DEFAULT '',
    google_review_url TEXT NOT NULL DEFAULT '',
    answer_mode TEXT NOT NULL DEFAULT 'after_hours',
    plan TEXT NOT NULL DEFAULT 'pro',
    status TEXT NOT NULL DEFAULT 'trial',
    trial_ends_at TEXT,
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    avg_job_value INTEGER NOT NULL DEFAULT 450,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY,
    shop_id INTEGER NOT NULL REFERENCES shops(id),
    call_sid TEXT UNIQUE NOT NULL,
    caller_phone TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'in_progress',
    outcome TEXT,
    is_emergency INTEGER NOT NULL DEFAULT 0,
    caller_name TEXT NOT NULL DEFAULT '',
    callback_phone TEXT NOT NULL DEFAULT '',
    address TEXT NOT NULL DEFAULT '',
    problem TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    messages TEXT NOT NULL DEFAULT '[]',
    transcript TEXT NOT NULL DEFAULT '[]',
    owner_notified INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY,
    shop_id INTEGER NOT NULL REFERENCES shops(id),
    call_id INTEGER REFERENCES calls(id),
    customer_name TEXT NOT NULL DEFAULT '',
    customer_phone TEXT NOT NULL DEFAULT '',
    address TEXT NOT NULL DEFAULT '',
    problem TEXT NOT NULL DEFAULT '',
    start_at TEXT NOT NULL,
    end_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'booked',
    external_event_id TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS quotes (
    id INTEGER PRIMARY KEY,
    shop_id INTEGER NOT NULL REFERENCES shops(id),
    customer_name TEXT NOT NULL DEFAULT '',
    customer_phone TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    amount REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS followups (
    id INTEGER PRIMARY KEY,
    shop_id INTEGER NOT NULL REFERENCES shops(id),
    kind TEXT NOT NULL,
    to_phone TEXT NOT NULL,
    body TEXT NOT NULL,
    send_at TEXT NOT NULL,
    sent_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    call_id INTEGER REFERENCES calls(id),
    appointment_id INTEGER REFERENCES appointments(id),
    quote_id INTEGER REFERENCES quotes(id)
);

CREATE TABLE IF NOT EXISTS opt_outs (
    shop_id INTEGER NOT NULL,
    phone TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (shop_id, phone)
);

CREATE TABLE IF NOT EXISTS sms_log (
    id INTEGER PRIMARY KEY,
    shop_id INTEGER,
    direction TEXT NOT NULL,
    to_phone TEXT NOT NULL,
    from_phone TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_calls_shop ON calls(shop_id, started_at);
CREATE INDEX IF NOT EXISTS idx_appts_shop ON appointments(shop_id, start_at);
CREATE INDEX IF NOT EXISTS idx_followups_due ON followups(status, send_at);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or get_settings().database_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(path: str | None = None) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert(conn: sqlite3.Connection, table: str, values: dict[str, Any]) -> int:
    cols = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    cur = conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(values.values()))
    return int(cur.lastrowid)


def update(conn: sqlite3.Connection, table: str, row_id: int, values: dict[str, Any]) -> None:
    sets = ", ".join(f"{k} = ?" for k in values)
    conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), row_id])


def get(conn: sqlite3.Connection, table: str, row_id: int) -> sqlite3.Row | None:
    return conn.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()


def shop_by_number(conn: sqlite3.Connection, number: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM shops WHERE twilio_number = ?", (number,)).fetchone()


def business_hours(shop: sqlite3.Row) -> dict[str, list[str]]:
    return json.loads(shop["business_hours"] or "{}")


DEFAULT_HOURS = {
    "mon": ["08:00", "17:00"],
    "tue": ["08:00", "17:00"],
    "wed": ["08:00", "17:00"],
    "thu": ["08:00", "17:00"],
    "fri": ["08:00", "17:00"],
}
