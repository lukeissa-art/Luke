"""Startup webhook sync, the call-loop guard, and owner phone verification."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import admin, db, phone_numbers, scheduling, sms, telephony
from app.config import get_settings
from app.main import app
from tests.conftest import make_shop
from tests.test_phone_numbers import FakeTwilio

AUTH = ("admin", "secret")
LINE = "+15125550199"


@pytest.fixture
def twilio_on(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://tradedesk.example.com")
    get_settings.cache_clear()
    fake = FakeTwilio(owned=("+15125550188",))
    monkeypatch.setattr(sms, "_twilio", lambda: fake)
    return fake


@pytest.fixture
def client(settings):
    telephony._ringing_owner.clear()
    telephony._accepted.clear()
    admin._verify_codes.clear()
    with TestClient(app) as c:
        yield c


# --- 1. Numbers follow the server to its new address -------------------------------------

def test_startup_sync_repoints_numbers_left_on_an_old_address(conn, twilio_on):
    moved = make_shop(conn, twilio_number="+15125550188")  # owned, still pointing at the old host
    make_shop(conn, name="Demo", twilio_number="+15125550199")  # not on the account: skipped
    twilio_on.owned["+15125550188"].voice_url = "https://tradedesk-ai-8mif.onrender.com/voice/incoming"
    assert phone_numbers.sync_all() == 1
    sid, kw = twilio_on.updated[0]
    assert sid == "PN0" and kw["voice_url"] == "https://tradedesk.example.com/voice/incoming"

    # Already correct: nothing to do the next time
    for key, value in phone_numbers.webhooks(db.get(conn, "shops", moved["id"])).items():
        setattr(twilio_on.owned["+15125550188"], key, value)
    assert phone_numbers.sync_all() == 0


def test_startup_sync_does_nothing_without_twilio(conn):
    make_shop(conn, twilio_number="+15125550188")
    assert phone_numbers.sync_all() == 0


# --- 2. Owner's cell forwarding back to the assistant can't loop -----------------------

def _incoming(client, sid, caller="+15125550142"):
    return client.post("/voice/incoming", data={"To": LINE, "From": caller, "CallSid": sid})


def test_ring_owner_bouncing_back_is_answered_busy(client, conn, monkeypatch):
    make_shop(conn, answer_mode="after_hours")
    monkeypatch.setattr(scheduling, "is_open_now", lambda shop, now=None: True)
    assert "<Dial" in _incoming(client, "CA1").text  # rings the owner's cell...
    # ...which forwards to the assistant: same caller, new call, while still ringing
    assert '<Reject reason="busy"/>' in _incoming(client, "CA2").text
    # The original call then falls through to the assistant
    r = client.post("/voice/no-answer", data={"To": LINE, "From": "+15125550142", "CallSid": "CA1",
                                              "DialCallStatus": "busy"})
    assert '<Gather input="speech"' in r.text
    # Once that ring is over, the same person calling again is handled normally
    assert '<Reject' not in _incoming(client, "CA3").text


def test_transfer_forwarded_back_to_the_assistant_is_answered_busy(client, conn):
    make_shop(conn, answer_mode="always")
    # A transfer's caller ID is the shop's own number, so a bounce arrives "from" our own line.
    assert '<Reject reason="busy"/>' in _incoming(client, "CA5", caller=LINE).text


# --- 3. Owner phone verification ------------------------------------------------------

@pytest.fixture
def texts(twilio_on):
    sent = []
    twilio_on.messages = SimpleNamespace(create=lambda **kw: sent.append(kw))
    return sent


def _code(texts):
    return texts[-1]["body"].rsplit(" ", 1)[1]


def test_owner_phone_verification(client, conn, texts):
    shop = make_shop(conn)
    page = client.get(f"/admin/shops/{shop['id']}", auth=AUTH).text
    assert "Owner&#39;s phone not verified" in page or "Owner's phone not verified" in page

    client.post(f"/admin/shops/{shop['id']}/owner/send-code", auth=AUTH)
    assert texts[0]["to"] == "+15125550100" and texts[0]["from_"] == LINE
    wrong = f"{(int(_code(texts)) + 1) % 1_000_000:06d}"
    r = client.post(f"/admin/shops/{shop['id']}/owner/verify", data={"code": wrong}, auth=AUTH,
                    follow_redirects=False)
    assert "doesn%27t+match" in r.headers["location"]
    client.post(f"/admin/shops/{shop['id']}/owner/verify", data={"code": _code(texts)}, auth=AUTH)
    assert db.get(conn, "shops", shop["id"])["owner_phone_verified"] == "+15125550100"
    assert ">verified</span>" in client.get(f"/admin/shops/{shop['id']}", auth=AUTH).text

    # Changing the owner's phone makes it unverified again
    conn.execute("UPDATE shops SET owner_phone = '+15125550177' WHERE id = ?", (shop["id"],))
    conn.commit()
    assert ">verified</span>" not in client.get(f"/admin/shops/{shop['id']}", auth=AUTH).text


def test_old_code_doesnt_verify_a_new_phone(client, conn, texts):
    shop = make_shop(conn)
    client.post(f"/admin/shops/{shop['id']}/owner/send-code", auth=AUTH)
    conn.execute("UPDATE shops SET owner_phone = '+15125550177' WHERE id = ?", (shop["id"],))
    conn.commit()
    client.post(f"/admin/shops/{shop['id']}/owner/verify", data={"code": _code(texts)}, auth=AUTH)
    assert db.get(conn, "shops", shop["id"])["owner_phone_verified"] is None


def test_verification_explains_when_texts_cant_be_sent(client, conn):
    shop = make_shop(conn)
    r = client.post(f"/admin/shops/{shop['id']}/owner/send-code", auth=AUTH, follow_redirects=False)
    assert "Can%27t+send+texts" in r.headers["location"]
