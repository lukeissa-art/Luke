"""Call lifecycle: start, monthly caps, and wrap-up (owner summary + follow-ups)."""

import sqlite3
from datetime import datetime

from . import db, followups, sms
from .plans import get_plan
from .receptionist import pretty_phone


def start_call(conn: sqlite3.Connection, shop: sqlite3.Row, call_sid: str, caller: str,
               now: datetime | None = None) -> sqlite3.Row:
    existing = conn.execute("SELECT * FROM calls WHERE call_sid = ?", (call_sid,)).fetchone()
    if existing:
        return existing
    call_id = db.insert(
        conn,
        "calls",
        {"shop_id": shop["id"], "call_sid": call_sid, "caller_phone": caller,
         "started_at": db.iso(now or db.utcnow())},
    )
    return db.get(conn, "calls", call_id)


def calls_this_month(conn: sqlite3.Connection, shop_id: int, now: datetime | None = None) -> int:
    now = now or db.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    row = conn.execute(
        "SELECT COUNT(*) FROM calls WHERE shop_id = ? AND started_at >= ? AND call_sid NOT LIKE 'demo-%'",
        (shop_id, db.iso(month_start)),
    ).fetchone()
    return int(row[0])


def over_cap(conn: sqlite3.Connection, shop: sqlite3.Row, now: datetime | None = None) -> bool:
    return calls_this_month(conn, shop["id"], now) >= get_plan(shop["plan"]).monthly_call_cap


def finalize_call(conn: sqlite3.Connection, call_id: int, *, status: str = "completed",
                  now: datetime | None = None) -> None:
    """Idempotent: safe to call from both the hangup path and Twilio's status callback."""
    call = db.get(conn, "calls", call_id)
    if call is None or call["owner_notified"]:
        return
    shop = db.get(conn, "shops", call["shop_id"])
    now = now or db.utcnow()
    outcome = call["outcome"] or ("message_taken" if call["problem"] else "hangup")
    db.update(conn, "calls", call_id,
              {"ended_at": db.iso(now), "status": status, "outcome": outcome, "owner_notified": 1})

    if call["call_sid"].startswith("demo-"):
        # Demo/test calls: show the owner text, but free the slot and skip customer follow-ups.
        conn.execute("UPDATE appointments SET status = 'cancelled' WHERE call_id = ?", (call_id,))
        sms.send(conn, shop, shop["owner_phone"], "[TEST CALL] " + owner_summary(
            db.get(conn, "calls", call_id), outcome), to_owner=True)
        return
    if outcome in ("spam", "not_a_customer"):
        return
    if outcome == "hangup" and not call["caller_name"]:
        # Caller hung up before giving details: still worth a text-back.
        followups.schedule_unbooked(conn, shop, call, now=now)
        sms.send(conn, shop, shop["owner_phone"],
                 f"Missed caller {pretty_phone(call['caller_phone'])} hung up early. "
                 "We texted them to rebook.", to_owner=True)
        return

    sms.send(conn, shop, shop["owner_phone"], owner_summary(call, outcome), to_owner=True)
    if outcome not in ("booked", "transferred"):
        followups.schedule_unbooked(conn, shop, call, now=now)


def owner_summary(call: sqlite3.Row, outcome: str) -> str:
    label = {
        "booked": "BOOKED",
        "transferred": "TRANSFERRED TO YOU",
        "message_taken": "CALL BACK",
        "caller_done": "INFO CALL",
    }.get(outcome, outcome.upper())
    urgent = "URGENT " if call["is_emergency"] else ""
    lines = [f"{urgent}{label}: {call['caller_name'] or 'Unknown caller'} "
             f"{pretty_phone(call['callback_phone'] or call['caller_phone'])}"]
    if call["address"]:
        lines.append(call["address"])
    if call["problem"]:
        lines.append(call["problem"])
    if call["summary"] and call["summary"] != call["problem"]:
        lines.append(call["summary"])
    return "\n".join(lines)[:600]


def complete_appointment(conn: sqlite3.Connection, appt_id: int, now: datetime | None = None) -> None:
    """Owner marks a job done (dashboard or 'DONE' text) -> schedule the review request."""
    appt = db.get(conn, "appointments", appt_id)
    if appt is None or appt["status"] == "completed":
        return
    now = now or db.utcnow()
    db.update(conn, "appointments", appt_id, {"status": "completed", "completed_at": db.iso(now)})
    shop = db.get(conn, "shops", appt["shop_id"])
    followups.schedule_review_request(conn, shop, db.get(conn, "appointments", appt_id), now=now)
    if get_plan(shop["plan"]).maintenance_reminders:
        followups.schedule_maintenance(conn, shop, appt, now=now)
