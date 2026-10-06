"""Finding open slots and booking them.

Availability = the shop's business hours, cut into fixed-length arrival windows, minus
appointments already booked here, minus busy time on the shop's Google Calendar (if connected).
"""

import base64
import json
import logging
import sqlite3
from datetime import date, datetime, time, timedelta
from urllib.parse import quote
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
# One Google service account for the whole company. Each shop shares their calendar with the
# service account's email ("Make changes to events"); the shop's calendar ID goes on its setup
# page. Busy time on that calendar blocks slots, and jobs the assistant books are added to it.

CALENDAR_API = "https://www.googleapis.com/calendar/v3"
GOOGLE_TIMEOUT = 8  # seconds: this runs while a caller is waiting
_google_session = None


class CalendarError(Exception):
    """A problem worth showing on the dashboard."""


def _service_account_info() -> dict | None:
    s = get_settings()
    raw = s.google_service_account_json.strip()
    if raw:
        if not raw.startswith("{"):
            try:
                raw = base64.b64decode(raw).decode()
            except ValueError:
                pass
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise CalendarError("GOOGLE_SERVICE_ACCOUNT_JSON isn't valid JSON. Paste the whole "
                                "key file, from { to }.") from exc
    if s.google_service_account_file:
        with open(s.google_service_account_file) as f:
            return json.load(f)
    return None


def google_configured() -> bool:
    s = get_settings()
    return bool(s.google_service_account_json.strip() or s.google_service_account_file)


def service_account_email() -> str:
    try:
        info = _service_account_info()
    except (CalendarError, OSError, ValueError):
        return ""
    return (info or {}).get("client_email", "")


def _google():
    global _google_session
    if _google_session is None:
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2 import service_account

        info = _service_account_info()
        if not info:
            raise CalendarError("Google Calendar isn't set up: add GOOGLE_SERVICE_ACCOUNT_JSON to the "
                                "server's environment variables.")
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/calendar"])
        _google_session = AuthorizedSession(creds)
    return _google_session


def _call(method: str, path: str, **kwargs) -> dict:
    resp = _google().request(method, CALENDAR_API + path, timeout=GOOGLE_TIMEOUT, **kwargs)
    if resp.status_code >= 400:
        try:
            message = resp.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = resp.text[:200]
        raise CalendarError(f"Google Calendar said: {message} (HTTP {resp.status_code})")
    return resp.json() if resp.content else {}


def _freebusy(calendar_id: str, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    result = _call("POST", "/freeBusy", json={
        "timeMin": start.isoformat(), "timeMax": end.isoformat(), "items": [{"id": calendar_id}]})
    cal = result.get("calendars", {}).get(calendar_id, {})
    if cal.get("errors"):
        reason = cal["errors"][0].get("reason", "error")
        if reason == "notFound":
            raise CalendarError(f"Can't see calendar {calendar_id}. Share it with "
                                f"{service_account_email() or 'the service account'} "
                                "(\"Make changes to events\") and check the calendar ID.")
        raise CalendarError(f"Google Calendar couldn't read {calendar_id}: {reason}")
    return [(db.parse_iso(p["start"]), db.parse_iso(p["end"])) for p in cal.get("busy", [])]


def google_busy(calendar_id: str, start: datetime, end: datetime):
    """Busy periods on the shop's calendar. On any failure, fall back to local bookings only."""
    try:
        return _freebusy(calendar_id, start, end)
    except Exception:
        log.exception("Google free/busy lookup failed for %s; using local bookings only", calendar_id)
        return []


def google_create_event(calendar_id, start, end, *, summary, description, location) -> str | None:
    try:
        event = _call("POST", f"/calendars/{quote(calendar_id, safe='')}/events", json={
            "summary": summary,
            "description": description,
            "location": location,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
        })
        return event.get("id")
    except Exception:
        log.exception("Google event insert failed for %s; booking kept locally", calendar_id)
        return None


def google_delete_event(calendar_id: str, event_id: str) -> None:
    _call("DELETE", f"/calendars/{quote(calendar_id, safe='')}/events/{quote(event_id, safe='')}")


def google_check(calendar_id: str, now: datetime | None = None) -> str:
    """Prove we can read busy times and add events. Returns a summary; raises CalendarError."""
    now = now or db.utcnow()
    busy = _freebusy(calendar_id, now, now + timedelta(days=7))
    start = now + timedelta(days=30)
    try:
        event = _call("POST", f"/calendars/{quote(calendar_id, safe='')}/events", json={
            "summary": f"{get_settings().company_name} connection test (deleted automatically)",
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": (start + timedelta(minutes=15)).isoformat()},
        })
    except CalendarError as exc:
        raise CalendarError(f"Can read the calendar but can't add jobs to it. Change the sharing "
                            f"setting to \"Make changes to events\". ({exc})") from exc
    google_delete_event(calendar_id, event["id"])
    return f"Connected. {len(busy)} busy block(s) in the next 7 days will be skipped when booking."
