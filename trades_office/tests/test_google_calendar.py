import base64
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import db, scheduling
from app.config import get_settings
from app.main import app
from tests.conftest import NOW, make_shop

AUTH = ("admin", "secret")
CAL = "owner@gmail.com"
KEY = {"type": "service_account", "client_email": "tradedesk@tradedesk-cal.iam.gserviceaccount.com"}


class FakeGoogle:
    """Stands in for google-auth's AuthorizedSession talking to the Calendar REST API."""

    def __init__(self, busy=(), freebusy_error=None, insert_status=200):
        self.busy, self.freebusy_error, self.insert_status = list(busy), freebusy_error, insert_status
        self.requests, self.events, self.deleted = [], [], []

    def request(self, method, url, timeout=None, json=None):
        assert timeout, "Calendar calls must have a timeout: a caller is waiting"
        self.requests.append((method, url, json))
        if url.endswith("/freeBusy"):
            cal = {"busy": self.busy}
            if self.freebusy_error:
                cal = {"busy": [], "errors": [{"domain": "global", "reason": self.freebusy_error}]}
            return _resp(200, {"calendars": {json["items"][0]["id"]: cal}})
        if method == "POST" and url.endswith("/events"):
            if self.insert_status >= 400:
                return _resp(self.insert_status, {"error": {"message": "Forbidden"}})
            self.events.append(json)
            return _resp(200, {"id": f"evt{len(self.events)}"})
        if method == "DELETE":
            self.deleted.append(url.rsplit("/", 1)[1])
            return _resp(204, None)
        raise AssertionError(f"unexpected request {method} {url}")


def _resp(status, body):
    content = b"" if body is None else json.dumps(body).encode()
    return SimpleNamespace(status_code=status, content=content, text=content.decode(), json=lambda: body)


@pytest.fixture
def google(monkeypatch):
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", json.dumps(KEY))
    get_settings.cache_clear()
    fake = FakeGoogle()
    monkeypatch.setattr(scheduling, "_google", lambda: fake)
    return fake


@pytest.fixture
def client(settings):
    with TestClient(app) as c:
        yield c


def _cal_shop(conn):
    return make_shop(conn, calendar_provider="google", google_calendar_id=CAL)


def test_busy_time_on_owners_calendar_blocks_slots(conn, google):
    shop = _cal_shop(conn)
    assert scheduling.open_slots(conn, shop, now=NOW)[0]["label"].startswith("Wednesday September 30, 12 PM")
    # Owner blocks 12:30-1:30 local (17:30-18:30 UTC) on their phone
    google.busy = [{"start": "2026-09-30T17:30:00Z", "end": "2026-09-30T18:30:00Z"}]
    assert scheduling.open_slots(conn, shop, now=NOW)[0]["label"].startswith("Wednesday September 30, 2 PM")


def test_booked_jobs_are_added_to_owners_calendar(conn, google):
    shop = _cal_shop(conn)
    slot = scheduling.open_slots(conn, shop, now=NOW)[0]
    appt_id = scheduling.book(conn, shop, start_iso=slot["start"], customer_name="Ann",
                              customer_phone="+15125550142", address="1 Main St", problem="leaking water heater",
                              now=NOW)
    event = google.events[0]
    assert event["summary"] == "leaking water heater - Ann"
    assert "+15125550142" in event["description"] and event["location"] == "1 Main St"
    assert "owner%40gmail.com/events" in google.requests[-1][1]
    assert db.get(conn, "appointments", appt_id)["external_event_id"] == "evt1"


def test_unshared_calendar_falls_back_to_local_bookings(conn, google):
    google.freebusy_error = "notFound"
    shop = _cal_shop(conn)
    assert scheduling.open_slots(conn, shop, now=NOW)  # still offers times from business hours
    with pytest.raises(scheduling.CalendarError, match="Share it with tradedesk@"):
        scheduling.google_check(CAL, now=NOW)


def test_connect_calendar_tests_then_saves(client, conn, google):
    shop = make_shop(conn)
    r = client.post(f"/admin/shops/{shop['id']}/calendar", data={"calendar_id": f" {CAL} "},
                    auth=AUTH, follow_redirects=False)
    assert "cal_ok" in r.headers["location"]
    assert google.deleted == ["evt1"]  # the test event is cleaned up
    saved = db.get(conn, "shops", shop["id"])
    assert saved["calendar_provider"] == "google" and saved["google_calendar_id"] == CAL
    page = client.get(r.headers["location"], auth=AUTH).text
    assert "Connected to <b>owner@gmail.com</b>" in page


def test_read_only_sharing_is_reported_and_not_saved(client, conn, google):
    google.insert_status = 403
    shop = make_shop(conn)
    r = client.post(f"/admin/shops/{shop['id']}/calendar", data={"calendar_id": CAL},
                    auth=AUTH, follow_redirects=False)
    page = client.get(r.headers["location"], auth=AUTH).text
    assert "Make changes to events" in page and "can&#39;t add jobs" in page
    assert db.get(conn, "shops", shop["id"])["calendar_provider"] != "google"


def test_disconnect(client, conn, google):
    shop = _cal_shop(conn)
    client.post(f"/admin/shops/{shop['id']}/calendar/disconnect", auth=AUTH)
    assert db.get(conn, "shops", shop["id"])["calendar_provider"] == "local"


def test_shop_page_shows_email_to_share_with(client, conn, google):
    shop = make_shop(conn)
    page = client.get(f"/admin/shops/{shop['id']}", auth=AUTH).text
    assert KEY["client_email"] in page and "Make changes to events" in page


def test_shop_page_explains_missing_setup(client, conn):
    shop = make_shop(conn)
    page = client.get(f"/admin/shops/{shop['id']}", auth=AUTH).text
    assert "GOOGLE_SERVICE_ACCOUNT_JSON" in page


@pytest.mark.parametrize("value", [json.dumps(KEY), base64.b64encode(json.dumps(KEY).encode()).decode()])
def test_service_account_key_from_environment(monkeypatch, value):
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", value)
    get_settings.cache_clear()
    assert scheduling.google_configured()
    assert scheduling.service_account_email() == KEY["client_email"]


def test_bad_service_account_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "not json")
    get_settings.cache_clear()
    assert scheduling.service_account_email() == ""
    with pytest.raises(scheduling.CalendarError, match="isn't valid JSON"):
        scheduling._service_account_info()
