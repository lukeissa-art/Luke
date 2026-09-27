"""Finding open slots and booking them.

Availability = the shop's business hours, cut into fixed-length arrival windows, minus
appointments already booked here, minus busy time on the shop's Google Calendar (if connected).
"""

import logging
import sqlite3
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from . import db
from .config import get_settings

log = logging.getLogger(__name__)

DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MIN_LEAD = timedelta(hours=2)


def _busy_local(conn: sqlite3.Connection, shop_id: int, start: datetime, end: datetime):
    rows = conn.execute(
        "SELECT start_at, end_at FROM appointments WHERE shop_id = ? AND status = 'booked' "
        "AND start_at < ? AND end_at > ?",
        (shop_id, db.iso(end), db.iso(start)),
    ).fetchall()
    return [(db.parse_iso(r["start_at"]), db.parse_iso(r["end_at"])) for r in rows]


def _overlaps(a_start, a_end, busy) -> bool:
    return any(a_start < b_end and a_end > b_start for b_start, b_end in busy)


def open_slots(
    conn: sqlite3.Connection,
    shop: sqlite3.Row,
    *,
    now: datetime | None = None,
    on_date: date | None = None,
    limit: int = 4,
    days_ahead: int = 10,
    min_lead: timedelta = MIN_LEAD,
) -> list[dict]:
    tz = ZoneInfo(shop["timezone"])
    now = (now or db.utcnow()).astimezone(tz)
    hours = db.business_hours(shop) or db.DEFAULT_HOURS
    length = timedelta(minutes=shop["slot_minutes"])

    start_day = on_date or now.date()
    window_start = datetime.combine(start_day, time.min, tz)
    window_end = window_start + timedelta(days=1 if on_date else days_ahead)

    busy = _busy_local(conn, shop["id"], window_start, window_end)
    if shop["calendar_provider"] == "google" and shop["google_calendar_id"]:
        busy += google_busy(shop["google_calendar_id"], window_start, window_end)

    slots: list[dict] = []
    day = start_day
    while datetime.combine(day, time.min, tz) < window_end and len(slots) < limit:
        span = hours.get(DAY_KEYS[day.weekday()])
        if span:
            open_t, close_t = (time.fromisoformat(t) for t in span)
            cursor = datetime.combine(day, open_t, tz)
            close = datetime.combine(day, close_t, tz)
            while cursor + length <= close and len(slots) < limit:
                end = cursor + length
                if cursor >= now + min_lead and not _overlaps(cursor, end, busy):
                    slots.append(
                        {
                            "start": db.iso(cursor),
                            "label": f"{cursor:%A %B} {cursor.day}, "
                            f"{_fmt_time(cursor)} to {_fmt_time(end)}",
                        }
                    )
                cursor = end
        day += timedelta(days=1)
    return slots


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0").replace(":00", "")


def label(shop: sqlite3.Row, start_iso: str) -> str:
    tz = ZoneInfo(shop["timezone"])
    start = db.parse_iso(start_iso).astimezone(tz)
    end = start + timedelta(minutes=shop["slot_minutes"])
    return f"{start:%A %B} {start.day}, {_fmt_time(start)} to {_fmt_time(end)}"


def is_open_now(shop: sqlite3.Row, now: datetime | None = None) -> bool:
    tz = ZoneInfo(shop["timezone"])
    now = (now or db.utcnow()).astimezone(tz)
    span = (db.business_hours(shop) or db.DEFAULT_HOURS).get(DAY_KEYS[now.weekday()])
    if not span:
        return False
    open_t, close_t = (time.fromisoformat(t) for t in span)
    return open_t <= now.time() < close_t


class SlotUnavailable(Exception):
    pass


def book(
    conn: sqlite3.Connection,
    shop: sqlite3.Row,
    *,
    start_iso: str,
    customer_name: str,
    customer_phone: str,
    address: str,
    problem: str,
    call_id: int | None = None,
    now: datetime | None = None,
    sync_external: bool = True,
) -> int:
    start = db.parse_iso(start_iso)
    end = start + timedelta(minutes=shop["slot_minutes"])
    local_day = start.astimezone(ZoneInfo(shop["timezone"])).date()
    # No lead-time check here: the window was offered with lead time a minute ago; just make
    # sure it's a real window, still free, and hasn't started.
    valid = {s["start"] for s in open_slots(conn, shop, now=now, on_date=local_day, limit=50,
                                            min_lead=timedelta(0))}
    if db.iso(start) not in valid:
        raise SlotUnavailable(start_iso)

    external_id = None
    if sync_external and shop["calendar_provider"] == "google" and shop["google_calendar_id"]:
        external_id = google_create_event(
            shop["google_calendar_id"],
            start,
            end,
            summary=f"{problem[:60]} - {customer_name}",
            description=f"Customer: {customer_name}\nPhone: {customer_phone}\n"
            f"Address: {address}\nProblem: {problem}\nBooked by {get_settings().company_name}",
            location=address,
        )

    return db.insert(
        conn,
        "appointments",
        {
            "shop_id": shop["id"],
            "call_id": call_id,
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "address": address,
            "problem": problem,
            "start_at": db.iso(start),
            "end_at": db.iso(end),
            "external_event_id": external_id,
            "created_at": db.iso(now or db.utcnow()),
        },
    )


# --- Google Calendar -------------------------------------------------------------------
# Optional: install google-api-python-client + google-auth, create a service account,
# and have each shop share their calendar with the service account's email.


def _google_service():
    from google.oauth2 import service_account  # type: ignore[import-not-found]
    from googleapiclient.discovery import build  # type: ignore[import-not-found]

    creds = service_account.Credentials.from_service_account_file(
        get_settings().google_service_account_file,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def google_busy(calendar_id: str, start: datetime, end: datetime):
    try:
        result = (
            _google_service()
            .freebusy()
            .query(
                body={
                    "timeMin": start.isoformat(),
                    "timeMax": end.isoformat(),
                    "items": [{"id": calendar_id}],
                }
            )
            .execute()
        )
    except Exception:
        log.exception("Google free/busy lookup failed for %s; using local bookings only", calendar_id)
        return []
    periods = result["calendars"][calendar_id]["busy"]
    return [(db.parse_iso(p["start"]), db.parse_iso(p["end"])) for p in periods]


def google_create_event(calendar_id, start, end, *, summary, description, location) -> str | None:
    try:
        event = (
            _google_service()
            .events()
            .insert(
                calendarId=calendar_id,
                body={
                    "summary": summary,
                    "description": description,
                    "location": location,
                    "start": {"dateTime": start.isoformat()},
                    "end": {"dateTime": end.isoformat()},
                },
            )
            .execute()
        )
        return event.get("id")
    except Exception:
        log.exception("Google event insert failed for %s; booking kept locally", calendar_id)
        return None
