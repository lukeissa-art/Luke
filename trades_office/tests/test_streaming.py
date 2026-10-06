import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import db, llm, receptionist, site
from app.config import get_settings
from app.main import app
from tests.conftest import FakeClaude, make_shop, reply, text, tool

AUTH = ("admin", "secret")
STREAM = {"Accept": "text/event-stream"}


@pytest.fixture
def client(settings):
    site._demo_hits.clear()
    with TestClient(app) as c:
        yield c


class FakeStream:
    def __init__(self, events, final):
        self.events, self.final = events, final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final


class FakeStreamingClaude(FakeClaude):
    """Like FakeClaude, but answers through messages.stream(), sending each text block in pieces."""

    def __init__(self, responses):
        super().__init__(responses)
        self.beta.messages.stream = self._stream

    def _stream(self, **kwargs):
        final = self._create(**kwargs)
        events = []
        for block in final.content:
            events.append(SimpleNamespace(type="content_block_start", content_block=block))
            if block.type == "text":
                words = block.text.split(" ")
                events += [SimpleNamespace(type="text", text=w + (" " if i < len(words) - 1 else ""))
                           for i, w in enumerate(words)]
        return FakeStream(events, final)


def events(r) -> list[dict]:
    assert r.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]


def streamed_text(evs: list[dict]) -> str:
    return "".join(e["text"] for e in evs if e["type"] == "text")


def test_text_sink_joins_separate_pieces_with_a_space():
    out = []
    sink = llm.TextSink(out.append)
    sink.segment()
    sink("Sure,")
    sink(" one moment.")
    sink.segment()
    sink("")
    sink("Tuesday works.")
    assert "".join(out) == "Sure, one moment. Tuesday works."


def test_claude_reply_is_streamed_piece_by_piece_across_tool_calls(conn, shop):
    fake = FakeStreamingClaude([
        reply(text("Let me check the schedule."), tool("check_availability", {"preferred_date": ""})),
        reply(text("I have Thursday morning open."), text("Does that work?")),
    ])
    call = db.insert(conn, "calls", {"shop_id": shop["id"], "call_sid": "web-abc", "caller_phone": "",
                                     "started_at": db.iso(db.utcnow())})
    pieces = []
    turn = receptionist.Receptionist(conn, shop, db.get(conn, "calls", call), client=fake).respond(
        "Can someone come this week?", pieces.append)
    assert len(pieces) > 3  # arrived in pieces, not all at once
    assert "".join(pieces) == turn.say == "Let me check the schedule. I have Thursday morning open. Does that work?"


def test_public_demo_streams_then_sends_the_final_reply(client, conn, monkeypatch):
    shop = make_shop(conn)
    monkeypatch.setenv("PUBLIC_DEMO_SHOP_ID", str(shop["id"]))
    get_settings.cache_clear()
    fake = FakeStreamingClaude([reply(text("Sorry to hear that. What's the address?"))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)

    evs = events(client.post("/demo/say", json={"text": "my sink is clogged"}, headers=STREAM))
    assert [e["type"] for e in evs[:-1]] == ["text"] * (len(evs) - 1) and len(evs) > 2
    done = evs[-1]
    assert done["type"] == "done" and done["status"] == 200 and done["action"] == "continue"
    assert streamed_text(evs) == done["say"] == "Sorry to hear that. What's the address?"
    assert db.get(conn, "calls", done["call_id"])["call_sid"].startswith("demo-web-")

    # Plain JSON still works for anything that doesn't ask for a stream.
    fake2 = FakeClaude([reply(text("Got it."))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake2)
    d = client.post("/demo/say", json={"text": "123 Main St", "call_id": done["call_id"]}).json()
    assert d["say"] == "Got it." and d["call_id"] == done["call_id"]


def test_streamed_errors_and_limits_still_come_back_as_json(client, conn, monkeypatch):
    assert client.post("/demo/say", json={"text": "hi"}, headers=STREAM).status_code == 404
    shop = make_shop(conn)
    monkeypatch.setenv("PUBLIC_DEMO_SHOP_ID", str(shop["id"]))
    get_settings.cache_clear()
    monkeypatch.setattr(site, "DEMO_MAX_MESSAGES_PER_HOUR", 0)
    r = client.post("/demo/say", json={"text": "hi"}, headers=STREAM)
    assert r.status_code == 429 and "limit" in r.json()["error"]


def test_ai_failure_mid_reply_sends_the_fallback_as_the_final_reply(client, conn, monkeypatch):
    shop = make_shop(conn)
    fake = FakeStreamingClaude([RuntimeError("overloaded")])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)
    evs = events(client.post(f"/chat/{shop['widget_key']}/say", json={"text": "hello"}, headers=STREAM))
    done = evs[-1]
    assert done["type"] == "done" and done["action"] == "transfer"
    # the page swaps in the final reply, which includes the shop's number to call
    assert "trouble" in done["say"] and "(512) 555-0199" in done["say"]


def test_website_chat_streams(client, conn, monkeypatch):
    shop = make_shop(conn)
    fake = FakeStreamingClaude([reply(text("Sorry to hear that! What's the best number to reach you?"))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)
    evs = events(client.post(f"/chat/{shop['widget_key']}/say", json={"text": "my sink is clogged"},
                             headers=STREAM))
    assert streamed_text(evs) == evs[-1]["say"] and evs[-1]["chat_id"]
    assert sum(e["type"] == "text" for e in evs) > 1


def test_admin_demo_streams(client, conn, monkeypatch):
    shop = make_shop(conn)
    fake = FakeStreamingClaude([reply(text("Thanks for calling. What's going on?"))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)
    evs = events(client.post(f"/admin/shops/{shop['id']}/demo/say", json={"text": "hi"}, headers=STREAM, auth=AUTH))
    assert streamed_text(evs) == evs[-1]["say"] == "Thanks for calling. What's going on?"


def test_non_streaming_client_still_streams_the_whole_reply_at_once(client, conn, monkeypatch):
    shop = make_shop(conn)
    fake = FakeClaude([reply(text("Hi there!"))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)
    evs = events(client.post(f"/chat/{shop['widget_key']}/say", json={"text": "hi"}, headers=STREAM))
    assert [e["type"] for e in evs] == ["text", "done"] and evs[0]["text"] == "Hi there!"
