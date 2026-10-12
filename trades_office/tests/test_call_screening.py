"""When the owner's phone is rung, they must press 1, so their voicemail can't swallow the call."""

import pytest
from fastapi.testclient import TestClient

from app import scheduling, telephony
from app.config import get_settings
from app.main import app
from tests.conftest import make_shop

SHOP_LINE = "+15125550199"


@pytest.fixture
def client(settings):
    telephony._accepted.clear()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ringing_owner(conn, monkeypatch):
    """A shop in after-hours mode during business hours: calls ring the owner first."""
    monkeypatch.setattr(scheduling, "is_open_now", lambda shop, now=None: True)
    return make_shop(conn, answer_mode="after_hours")


def _call(client, path, sid="CA1", **extra):
    return client.post(path, data={"To": SHOP_LINE, "From": "+15125550142", "CallSid": sid, **extra})


def test_owner_is_screened_before_being_connected(client, ringing_owner):
    r = _call(client, "/voice/incoming")
    assert f'/voice/screen?call=CA1&amp;shop={ringing_owner["id"]}' in r.text
    assert ">+15125550100</Number>" in r.text and "/voice/no-answer" in r.text

    r = client.post(f"/voice/screen?call=CA1&shop={ringing_owner['id']}", data={"CallSid": "CAowner"})
    assert '<Gather numDigits="1"' in r.text and "Press 1 to take it" in r.text
    assert "Riverside Plumbing" in r.text and "/voice/screen-result?call=CA1" in r.text


def test_voicemail_answering_sends_the_call_back_to_the_assistant(client, ringing_owner):
    _call(client, "/voice/incoming")
    # Voicemail picks up, never presses 1, the screen hangs up; Twilio reports the dial "completed".
    assert "<Hangup/>" in client.post("/voice/screen-result?call=CA1", data={"CallSid": "CAowner"}).text
    r = _call(client, "/voice/no-answer", DialCallStatus="completed")
    assert '<Gather input="speech"' in r.text and "Thanks for calling" in r.text


def test_owner_pressing_1_takes_the_call(client, ringing_owner):
    _call(client, "/voice/incoming")
    r = client.post("/voice/screen-result?call=CA1", data={"CallSid": "CAowner", "Digits": "1"})
    assert r.text.endswith("<Response></Response>")  # empty response = connect them
    # After they hang up, the caller isn't handed to the assistant
    assert "<Hangup/>" in _call(client, "/voice/no-answer", DialCallStatus="completed").text


def test_transfer_that_nobody_takes_tells_the_caller(client):
    assert "nobody picked up" in _call(client, "/voice/transfer-done", DialCallStatus="completed").text
    client.post("/voice/screen-result?call=CA9", data={"CallSid": "CAowner", "Digits": "1"})
    r = _call(client, "/voice/transfer-done", sid="CA9", DialCallStatus="completed")
    assert "nobody picked up" not in r.text and "<Hangup/>" in r.text


def test_transfer_from_the_assistant_is_screened(conn):
    shop = make_shop(conn)
    xml = telephony.dial_owner(shop, SHOP_LINE, "CA7")
    assert "/voice/transfer-done" in xml and "/voice/screen?call=CA7" in xml


def test_signature_check_covers_the_query_string(client, conn, monkeypatch):
    from twilio.request_validator import RequestValidator

    shop = make_shop(conn)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC1")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret-token")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://www.tradedeskuse.com")
    get_settings.cache_clear()
    path = f"/voice/screen?call=CA1&shop={shop['id']}"
    data = {"CallSid": "CAowner"}
    good = RequestValidator("secret-token").compute_signature("https://www.tradedeskuse.com" + path, data)
    assert client.post(path, data=data, headers={"X-Twilio-Signature": good}).status_code == 200
    assert client.post(path, data=data, headers={"X-Twilio-Signature": "forged"}).status_code == 403


def test_caller_who_hung_up_while_owner_rang_doesnt_start_a_conversation(client, conn, ringing_owner):
    _call(client, "/voice/incoming")
    r = _call(client, "/voice/no-answer", DialCallStatus="no-answer", CallStatus="completed")
    assert "<Hangup/>" in r.text and "<Gather" not in r.text
    assert conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0
