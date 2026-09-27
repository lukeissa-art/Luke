"""Automatic follow-up texts: unbooked callers, open quotes, review requests, maintenance.

Follow-ups are queued rows with a send time; `dispatch_due` (run every few minutes by the
worker) sends whatever is due. Queuing instead of sending inline makes them cancellable:
if a caller books before the text goes out, the text is cancelled.
"""

import sqlite3
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from . import db, sms
from .config import get_settings
from .plans import get_plan

# Customer-facing texts go out only between these local hours (TCPA-friendly, and polite).
QUIET_START = time(20, 0)
QUIET_END = time(8, 0)


def _queue(conn, shop, kind, to_phone, body, send_at, **refs) -> int | None:
    if not to_phone:
        return None
    return db.insert(
        conn,
        "followups",
        {"shop_id": shop["id"], "kind": kind, "to_phone": to_phone, "body": body,
         "send_at": db.iso(send_at), **refs},
    )


def next_allowed_time(shop: sqlite3.Row, when: datetime) -> datetime:
    """Push a send time out of quiet hours into the next morning."""
    tz = ZoneInfo(shop["timezone"])
    local = when.astimezone(tz)
    if local.time() >= QUIET_START:
        local = datetime.combine(local.date() + timedelta(days=1), QUIET_END, tz)
    elif local.time() < QUIET_END:
        local = datetime.combine(local.date(), QUIET_END, tz)
    return local


def schedule_unbooked(conn, shop, call, *, now: datetime) -> int | None:
    plan = get_plan(shop["plan"])
    if not plan.followups:
        return None
    send_at = next_allowed_time(shop, now + timedelta(minutes=get_settings().unbooked_followup_minutes))
    body = (
        f"Hi{(' ' + call['caller_name'].split()[0]) if call['caller_name'] else ''}, this is "
        f"{shop['name']}. Sorry we couldn't get you on the schedule during your call. "
        "Reply with a good time and we'll get you booked, or call us back anytime. Reply STOP to opt out."
    )
    return _queue(conn, shop, "unbooked_caller", call["callback_phone"] or call["caller_phone"], body,
                  send_at, call_id=call["id"])


def schedule_review_request(conn, shop, appt, *, now: datetime) -> int | None:
    if not get_plan(shop["plan"]).review_requests or not shop["google_review_url"]:
        return None
    send_at = next_allowed_time(shop, now + timedelta(hours=get_settings().review_request_hours))
    first = appt["customer_name"].split()[0] if appt["customer_name"] else "there"
    body = (
        f"Hi {first}, thanks for choosing {shop['name']}! If we did a good job, a quick Google review "
        f"helps a small local business a lot: {shop['google_review_url']} Reply STOP to opt out."
    )
    return _queue(conn, shop, "review_request", appt["customer_phone"], body, send_at,
                  appointment_id=appt["id"])


def schedule_quote_followups(conn, shop, quote, *, now: datetime) -> list[int]:
    """Two nudges on an open quote: after N days, and a week after that."""
    if not get_plan(shop["plan"]).quote_followups:
        return []
    days = get_settings().quote_followup_days
    first = quote["customer_name"].split()[0] if quote["customer_name"] else "there"
    texts = [
        (days, f"Hi {first}, it's {shop['name']}. Just checking whether you had any questions about "
               f"the estimate for {quote['description']}. Reply here and we'll get you on the schedule."),
        (days + 7, f"Hi {first}, {shop['name']} here. Your estimate for {quote['description']} is still "
                   "good. Want us to lock in a time? Reply STOP to opt out."),
    ]
    ids = []
    for offset, body in texts:
        fid = _queue(conn, shop, "quote", quote["customer_phone"], body,
                     next_allowed_time(shop, now + timedelta(days=offset)), quote_id=quote["id"])
        if fid:
            ids.append(fid)
    return ids


MAINTENANCE = {
    "hvac": (180, "it's been about six months since your service. Time to book a seasonal tune-up "
                  "before the rush?"),
    "plumbing": (365, "it's been about a year since your service. A yearly water heater flush adds years "
                      "to its life. Want to book one?"),
    "electrical": (365, "it's been about a year since your service. Want to book a quick safety check "
                        "of your panel?"),
}


def schedule_maintenance(conn, shop, appt, *, now: datetime) -> int | None:
    days, text = MAINTENANCE.get(shop["trade"], MAINTENANCE["plumbing"])
    first = appt["customer_name"].split()[0] if appt["customer_name"] else "there"
    body = f"Hi {first}, it's {shop['name']}: {text} Reply STOP to opt out."
    return _queue(conn, shop, "maintenance", appt["customer_phone"], body,
                  next_allowed_time(shop, now + timedelta(days=days)), appointment_id=appt["id"])


def cancel_for_phone(conn, shop_id: int, phone: str, kinds: tuple[str, ...]) -> None:
    marks = ",".join("?" for _ in kinds)
    conn.execute(
        f"UPDATE followups SET status = 'cancelled' WHERE shop_id = ? AND to_phone = ? "
        f"AND status = 'pending' AND kind IN ({marks})",
        (shop_id, phone, *kinds),
    )


def _still_relevant(conn, f) -> bool:
    if f["kind"] == "unbooked_caller" and f["call_id"]:
        call = db.get(conn, "calls", f["call_id"])
        booked_since = conn.execute(
            "SELECT 1 FROM appointments WHERE shop_id = ? AND customer_phone = ? AND created_at >= ? "
            "AND status != 'cancelled'",
            (f["shop_id"], f["to_phone"], call["started_at"]),
        ).fetchone()
        return booked_since is None
    if f["kind"] == "quote" and f["quote_id"]:
        return db.get(conn, "quotes", f["quote_id"])["status"] == "open"
    return True


def dispatch_due(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    now = now or db.utcnow()
    due = conn.execute(
        "SELECT * FROM followups WHERE status = 'pending' AND send_at <= ? ORDER BY send_at",
        (db.iso(now),),
    ).fetchall()
    sent = 0
    for f in due:
        shop = db.get(conn, "shops", f["shop_id"])
        if shop["status"] not in ("trial", "active") or not _still_relevant(conn, f):
            db.update(conn, "followups", f["id"], {"status": "skipped"})
            continue
        ok = sms.send(conn, shop, f["to_phone"], f["body"])
        db.update(conn, "followups", f["id"],
                  {"status": "sent" if ok else "skipped", "sent_at": db.iso(now)})
        sent += ok
    return sent
