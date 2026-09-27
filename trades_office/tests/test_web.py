import pytest
from fastapi.testclient import TestClient

from app import db, scheduling
from app.main import app
from tests.conftest import make_shop

AUTH = ("admin", "secret")


@pytest.fixture
def client(settings):
    with TestClient(app) as c:
        yield c


def test_incoming_call_is_answered_with_greeting(client, conn):
    make_shop(conn, answer_mode="always")
    r = client.post("/voice/incoming", data={"To": "+15125550199", "From": "+15125550142", "CallSid": "CA1"})
    assert r.status_code == 200
    assert "<Gather" in r.text and "Thanks for calling Riverside Plumbing" in r.text
    assert conn.execute("SELECT caller_phone FROM calls").fetchone()[0] == "+15125550142"


def test_after_hours_mode_rings_owner_first_during_business_hours(client, conn, monkeypatch):
    make_shop(conn, answer_mode="after_hours")
    monkeypatch.setattr(scheduling, "is_open_now", lambda shop, now=None: True)
    r = client.post("/voice/incoming", data={"To": "+15125550199", "From": "+1512", "CallSid": "CA2"})
    assert "<Dial" in r.text and "/voice/no-answer" in r.text
    r = client.post("/voice/no-answer", data={"To": "+15125550199", "From": "+1512", "CallSid": "CA2",
                                              "DialCallStatus": "no-answer"})
    assert "<Gather" in r.text


def test_paused_shop_passes_calls_through(client, conn):
    make_shop(conn, status="paused")
    r = client.post("/voice/incoming", data={"To": "+15125550199", "From": "+1512", "CallSid": "CA3"})
    assert "<Dial>+15125550100</Dial>" in r.text


def test_unknown_number(client):
    r = client.post("/voice/incoming", data={"To": "+19999999999", "From": "+1", "CallSid": "CA4"})
    assert "not in service" in r.text


def test_customer_stop_opts_out(client, conn):
    shop = make_shop(conn)
    client.post("/sms/incoming", data={"To": "+15125550199", "From": "+15125550142", "Body": "STOP"})
    assert conn.execute("SELECT COUNT(*) FROM opt_outs WHERE shop_id = ?", (shop["id"],)).fetchone()[0] == 1


def test_customer_reply_is_forwarded_to_owner(client, conn):
    make_shop(conn)
    r = client.post("/sms/incoming", data={"To": "+15125550199", "From": "+15125550142", "Body": "Tuesday works"})
    assert "get back to you" in r.text
    fwd = conn.execute("SELECT body FROM sms_log WHERE direction = 'outbound'").fetchone()
    assert "Tuesday works" in fwd["body"]


def test_owner_done_command(client, conn):
    shop = make_shop(conn)
    slot = scheduling.open_slots(conn, shop)[0]
    appt_id = scheduling.book(conn, shop, start_iso=slot["start"], customer_name="Ann", customer_phone="+15125550142",
                              address="1 Elm", problem="Leak")
    conn.commit()
    r = client.post("/sms/incoming", data={"To": "+15125550199", "From": "+15125550100", "Body": f"DONE {appt_id}"})
    assert "Review request queued" in r.text
    assert conn.execute("SELECT kind FROM followups").fetchone()["kind"] == "review_request"


def test_dashboard_requires_login(client):
    assert client.get("/admin/").status_code == 401


def test_onboard_shop_and_view_pages(client, conn):
    form = {"name": "Cool Air HVAC", "trade": "hvac", "owner_phone": "512 555 0111", "twilio_number": "5125550112",
            "timezone": "America/Chicago", "plan": "pro", "answer_mode": "always", "mon_open": "07:00",
            "mon_close": "18:00", "slot_minutes": "120", "avg_job_value": "600"}
    r = client.post("/admin/shops", data=form, auth=AUTH, follow_redirects=False)
    assert r.status_code == 303
    shop = conn.execute("SELECT * FROM shops WHERE name = 'Cool Air HVAC'").fetchone()
    assert shop["owner_phone"] == "+15125550111" and shop["status"] == "trial"
    assert db.business_hours(shop) == {"mon": ["07:00", "18:00"]}

    assert "Cool Air HVAC" in client.get("/admin/", auth=AUTH).text
    page = client.get(f"/admin/shops/{shop['id']}", auth=AUTH)
    assert page.status_code == 200 and "Recent calls" in page.text
    assert client.get(f"/admin/shops/{shop['id']}/edit", auth=AUTH).status_code == 200
    assert client.get(f"/admin/shops/{shop['id']}/demo", auth=AUTH).status_code == 200


def test_browser_demo_call(client, conn, monkeypatch):
    from app import receptionist
    from tests.conftest import FakeClaude, reply, text

    shop = make_shop(conn)
    fake = FakeClaude([reply(text("Sorry to hear that. What's your name?"))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)
    r = client.post(f"/admin/shops/{shop['id']}/demo/say", json={"text": "my toilet is clogged"}, auth=AUTH)
    d = r.json()
    assert d["action"] == "continue" and "name" in d["say"]
    call = db.get(conn, "calls", d["call_id"])
    assert call["call_sid"].startswith("demo-")
