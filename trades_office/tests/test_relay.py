"""Natural phone conversations over Twilio ConversationRelay (VOICE_ENGINE=relay)."""

import json
import threading

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import calls, db, receptionist, relay
from app.config import get_settings
from app.main import app
from tests.conftest import FakeClaude, make_shop, reply, text, tool

LINE = "+15125550199"
SID = "CArelay1"


@pytest.fixture
def relay_on(monkeypatch):
    monkeypatch.setenv("VOICE_ENGINE", "relay")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://www.tradedeskuse.com")
    get_settings.cache_clear()
    monkeypatch.setattr(relay, "_speaking_time", lambda text: 0)


@pytest.fixture
def client(settings):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def live_call(conn):
    shop = make_shop(conn, answer_mode="always")
    calls.start_call(conn, shop, SID, "+15155550142")
    conn.commit()
    return shop


def _connect(client, sid=SID):
    ws = client.websocket_connect(f"/voice/relay?call={sid}&token={relay._token(sid)}")
    return ws


def _say(ws, words):
    ws.send_text(json.dumps({"type": "prompt", "voicePrompt": words, "lang": "en-US", "last": True}))


def _until_last(ws):
    """Collect one reply: text tokens up to the one marked last."""
    tokens = []
    while True:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "text"
        tokens.append(msg["token"])
        if msg["last"]:
            return "".join(tokens)


def test_incoming_call_starts_a_conversation_relay(client, conn, relay_on):
    make_shop(conn, answer_mode="always")
    r = client.post("/voice/incoming", data={"To": LINE, "From": "+15155550142", "CallSid": SID})
    assert '<Connect action="https://www.tradedeskuse.com/voice/relay-done">' in r.text
    assert f'url="wss://www.tradedeskuse.com/voice/relay?call={SID}&amp;token={relay._token(SID)}"' in r.text
    assert 'welcomeGreeting="Thanks for calling Riverside Plumbing.' in r.text and "<Gather" not in r.text
    assert conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 1


def test_classic_line_is_still_the_default(client, conn):
    make_shop(conn, answer_mode="always")
    r = client.post("/voice/incoming", data={"To": LINE, "From": "+15155550142", "CallSid": SID})
    assert "<Gather" in r.text and "ConversationRelay" not in r.text


def test_optional_voice_settings(conn, relay_on, monkeypatch):
    monkeypatch.setenv("RELAY_TTS_PROVIDER", "ElevenLabs")
    monkeypatch.setenv("RELAY_VOICE", "Amelia")
    get_settings.cache_clear()
    xml = relay.relay_twiml(make_shop(conn), SID)
    assert 'ttsProvider="ElevenLabs"' in xml and 'voice="Amelia"' in xml


def test_conversation_turn_streams_the_reply(client, live_call, relay_on, monkeypatch):
    fake = FakeClaude([reply(text("Sorry about the leak! What's the address?"))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)
    with _connect(client) as ws:
        ws.send_text(json.dumps({"type": "setup", "callSid": SID, "from": "+15155550142", "to": LINE}))
        _say(ws, "my sink is leaking")
        assert _until_last(ws) == "Sorry about the leak! What's the address?"
    with db.session() as c:
        t = json.loads(c.execute("SELECT transcript FROM calls WHERE call_sid=?", (SID,)).fetchone()[0])
    assert t[0] == {"role": "caller", "text": "my sink is leaking"}


def test_goodbye_ends_the_session_with_a_hangup(client, live_call, relay_on, monkeypatch):
    fake = FakeClaude([reply(text("Thanks for calling, goodbye!"),
                             tool("end_call", {"outcome": "caller_done", "summary": "Asked about hours."}))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)
    with _connect(client) as ws:
        _say(ws, "that's all, thanks")
        assert "goodbye" in _until_last(ws)
        end = json.loads(ws.receive_text())
    assert end == {"type": "end", "handoffData": json.dumps({"action": "hangup"})}
    with db.session() as c:
        assert c.execute("SELECT owner_notified FROM calls WHERE call_sid=?", (SID,)).fetchone()[0] == 1


def test_gas_leak_script_is_spoken_then_transferred(client, live_call, relay_on, monkeypatch):
    monkeypatch.setattr(receptionist, "_client", lambda: FakeClaude([]))  # no AI involved
    with _connect(client) as ws:
        _say(ws, "I smell gas in my kitchen")
        assert "gas" in _until_last(ws).lower()
        end = json.loads(ws.receive_text())
    assert json.loads(end["handoffData"]) == {"action": "transfer"}


def test_interrupted_reply_is_not_sent(client, live_call, relay_on, monkeypatch):
    release = threading.Event()

    class SlowClaude(FakeClaude):
        def _create(self, **kwargs):
            release.wait(5)
            return super()._create(**kwargs)

    fake = SlowClaude([reply(text("A long answer the caller talks over")), reply(text("Got it, what's your name?"))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)
    with _connect(client) as ws:
        _say(ws, "hi I need")
        ws.send_text(json.dumps({"type": "interrupt", "utteranceUntilInterrupt": "", "durationUntilInterruptMs": 0}))
        _say(ws, "actually I need a plumber tomorrow")
        release.set()
        # The first reply was interrupted and dropped; the caller hears only the answer to what they said next.
        assert _until_last(ws) == "Got it, what's your name?"


def test_bad_token_is_refused(client, live_call, relay_on):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/voice/relay?call={SID}&token=forged") as ws:
            ws.receive_text()


def _done(client, **extra):
    return client.post("/voice/relay-done", data={"To": LINE, "From": "+15155550142", "CallSid": SID, **extra})


def test_relay_done_transfers_or_hangs_up(client, live_call, relay_on):
    r = _done(client, HandoffData=json.dumps({"action": "transfer"}))
    assert "<Dial" in r.text and "/voice/screen?call=" in r.text
    assert "<Hangup/>" in _done(client, HandoffData=json.dumps({"action": "hangup"})).text
    assert "<Hangup/>" in _done(client, CallStatus="completed").text


def test_relay_failure_never_drops_the_caller(client, live_call, relay_on):
    r = _done(client, SessionStatus="failed")
    assert "having trouble" in r.text and "<Dial" in r.text
