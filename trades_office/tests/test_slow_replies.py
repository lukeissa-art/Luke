"""A slow AI turn: the caller hears "one moment" instead of Twilio timing out the webhook."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import calls, receptionist, telephony
from app.main import app
from tests.conftest import FakeClaude, make_shop, reply, text

LINE = "+15125550199"
CALL = {"To": LINE, "From": "+15125550142", "CallSid": "CAslow"}


class SlowClaude(FakeClaude):
    """Answers only after `release` is set, like a model call that takes a while."""

    def __init__(self, responses):
        super().__init__(responses)
        self.release = threading.Event()

    def _create(self, **kwargs):
        self.release.wait(5)
        return super()._create(**kwargs)


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr(telephony, "REPLY_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(telephony, "HOLD_WAIT_SECONDS", 0.2)
    telephony._pending.clear()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def live_call(conn):
    shop = make_shop(conn, answer_mode="always")
    calls.start_call(conn, shop, CALL["CallSid"], CALL["From"])
    conn.commit()
    return shop


def test_slow_reply_says_one_moment_then_delivers(client, live_call, monkeypatch):
    slow = SlowClaude([reply(text("We can come Monday morning. Does that work?"))])
    monkeypatch.setattr(receptionist, "_client", lambda: slow)
    r = client.post("/voice/respond", data={**CALL, "SpeechResult": "Can someone come Monday?"})
    assert "One moment." in r.text and "/voice/hold" in r.text

    r = client.post("/voice/hold", data=CALL)  # still thinking: keep holding, quietly
    assert "/voice/hold" in r.text and "One moment" not in r.text

    slow.release.set()
    r = client.post("/voice/hold", data=CALL)
    assert "Monday morning" in r.text and '<Gather input="speech"' in r.text
    assert CALL["CallSid"] not in telephony._pending


def test_reply_that_never_comes_goes_to_the_owner(client, live_call, monkeypatch):
    slow = SlowClaude([reply(text("too late"))])
    monkeypatch.setattr(receptionist, "_client", lambda: slow)
    monkeypatch.setattr(telephony, "MAX_HOLD_SECONDS", 0.3)
    client.post("/voice/respond", data={**CALL, "SpeechResult": "hello?"})
    time.sleep(0.35)
    r = client.post("/voice/hold", data=CALL)
    assert "having trouble" in r.text and "/voice/transfer-done" in r.text
    slow.release.set()


def test_caller_hanging_up_while_on_hold_clears_the_reply(client, live_call, monkeypatch):
    slow = SlowClaude([reply(text("anyone there?"))])
    monkeypatch.setattr(receptionist, "_client", lambda: slow)
    client.post("/voice/respond", data={**CALL, "SpeechResult": "hi"})
    client.post("/voice/status", data={**CALL, "CallStatus": "completed"})
    assert "<Hangup/>" in client.post("/voice/hold", data=CALL).text
    slow.release.set()


def test_quick_reply_needs_no_hold(client, live_call, monkeypatch):
    monkeypatch.setattr(receptionist, "_client", lambda: FakeClaude([reply(text("Sure, what's your name?"))]))
    r = client.post("/voice/respond", data={**CALL, "SpeechResult": "I need a plumber"})
    assert "what&#x27;s your name" in r.text or "what's your name" in r.text
    assert "One moment" not in r.text


def test_turn_ends_after_one_second_of_silence():
    assert 'speechTimeout="1"' in telephony.gather("Hi")
