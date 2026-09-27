import json
from types import SimpleNamespace

from google.genai import types

from app import calls, db, llm, scheduling
from app.receptionist import Receptionist, make_backend
from tests.conftest import NOW


class FakeGemini:
    """Stands in for google.genai.Client: returns scripted responses, records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": [c.model_dump(mode="json", exclude_none=True) for c in contents],
                              "config": config})
        return self.responses.pop(0)


def gem(*parts, finish="STOP"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=list(parts)), finish_reason=finish)])


def text(t, signature=None):
    return types.Part(text=t, thought_signature=signature)


def call(name, args, id_=None):
    return types.Part(function_call=types.FunctionCall(name=name, args=args, id=id_))


def respond(conn, shop, call_id, fake, utterance):
    row = db.get(conn, "calls", call_id)
    return Receptionist(conn, shop, row, backend=llm.GeminiBackend(fake), now=NOW).respond(utterance)


def test_provider_selection(monkeypatch, settings):
    from app.config import get_settings

    assert llm.provider() == "anthropic"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()
    assert llm.provider() == "gemini"
    assert isinstance(make_backend(), llm.GeminiBackend)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    get_settings.cache_clear()
    assert llm.provider() == "anthropic"


def test_gemini_booking_call(conn, shop):
    c = calls.start_call(conn, shop, "CAG", "+15125550142", now=NOW)
    slot = scheduling.open_slots(conn, shop, now=NOW)[0]
    fake = FakeGemini([
        gem(call("check_availability", {"preferred_date": ""}), ),
        gem(text("I can get someone out today between 12 and 2. Does that work?", signature=b"sig1")),
        gem(call("book_appointment", {"slot_start": slot["start"], "customer_name": "Ann Lee",
                                      "callback_phone": "5125550142", "address": "12 Oak St",
                                      "problem_summary": "Leaking water heater"})),
        gem(text("You're booked. Goodbye!"), call("end_call", {"outcome": "booked", "summary": "Booked leak."})),
    ])
    t1 = respond(conn, shop, c["id"], fake, "My water heater is leaking")
    assert t1.action == "continue" and "12 and 2" in t1.say
    t2 = respond(conn, shop, c["id"], fake, "Yes please, Ann Lee, 12 Oak St")
    assert t2.action == "hangup" and "booked" in t2.say

    assert conn.execute("SELECT customer_name FROM appointments").fetchone()[0] == "Ann Lee"
    assert db.get(conn, "calls", c["id"])["outcome"] == "booked"

    req = fake.requests[0]
    assert req["model"] == "gemini-3.5-flash"
    names = {d.name for d in req["config"].tools[0].function_declarations}
    assert {"check_availability", "book_appointment", "end_call"} <= names
    # tool result goes back as a function_response with the parsed JSON result
    fr = fake.requests[1]["contents"][-1]["parts"][0]["function_response"]
    assert fr["name"] == "check_availability" and "slots" in fr["response"]["result"]
    # history survives the round trip through the database, thought signature included
    later = fake.requests[2]["contents"]
    assert later[0]["parts"][0]["text"].startswith("[Call started")
    assert any(p.get("thought_signature") for m in later for p in m["parts"])
    stored = json.loads(db.get(conn, "calls", c["id"])["messages"])
    assert all(m["role"] in ("user", "model") for m in stored)


def test_gemini_safety_block_takes_message(conn, shop):
    c = calls.start_call(conn, shop, "CAG2", "+15125550142", now=NOW)
    blocked = types.GenerateContentResponse(candidates=[], prompt_feedback=types.GenerateContentResponsePromptFeedback(
        block_reason="SAFETY"))
    turn = respond(conn, shop, c["id"], FakeGemini([blocked]), "something odd")
    assert turn.action == "hangup" and "call you back" in turn.say


def test_gemini_error_transfers(conn, shop):
    c = calls.start_call(conn, shop, "CAG3", "+15125550142", now=NOW)

    class Broken:
        models = SimpleNamespace(generate_content=lambda **kw: (_ for _ in ()).throw(RuntimeError("bad key")))

    turn = respond(conn, shop, c["id"], Broken(), "hello")
    assert turn.action == "transfer"


def test_gemini_empty_reply_is_an_error_not_a_refusal(conn, shop):
    # e.g. thinking used up the output budget: log it and transfer, don't hang up on the caller
    c = calls.start_call(conn, shop, "CAG4", "+15125550142", now=NOW)
    empty = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[]), finish_reason="MAX_TOKENS")])
    fake = FakeGemini([empty])
    turn = respond(conn, shop, c["id"], fake, "hello")
    assert turn.action == "transfer"
    cfg = fake.requests[0]["config"]
    assert cfg.max_output_tokens == 8192 and cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW


class NotFound(Exception):
    code = 404


def test_gemini_falls_back_when_model_unavailable(conn, shop, monkeypatch):
    monkeypatch.setattr(llm.GeminiBackend, "_model_override", None)
    c = calls.start_call(conn, shop, "CAG5", "+15125550142", now=NOW)

    class Picky(FakeGemini):
        def _generate(self, *, model, contents, config):
            if model != llm.FALLBACK_GEMINI_MODEL:
                self.requests.append({"model": model})
                raise NotFound("404 NOT_FOUND models/gemini-3.5-flash is not found")
            return super()._generate(model=model, contents=contents, config=config)

    fake = Picky([gem(text("Hi! How can I help?"))])
    turn = respond(conn, shop, c["id"], fake, "hello")
    assert turn.action == "continue" and "help" in turn.say
    assert [r["model"] for r in fake.requests] == ["gemini-3.5-flash", "gemini-2.5-flash"]
    decl = fake.requests[1]["config"].tools[0].function_declarations[0]
    assert "additionalProperties" not in json.dumps(decl.parameters_json_schema)
    assert fake.requests[1]["config"].thinking_config is None  # 2.5 doesn't take thinking_level
    assert llm.current_model() == "gemini-3.5-flash" or llm.provider() == "anthropic"


def test_ai_check_page_shows_the_error(settings, monkeypatch):
    from fastapi.testclient import TestClient

    from app import receptionist
    from app.main import app

    class Broken:
        def user_message(self, t):
            return {}

        def step(self, *a):
            raise RuntimeError("400 API key not valid. Please pass a valid API key.")

    monkeypatch.setattr(receptionist, "make_backend", lambda client=None: Broken())
    with TestClient(app) as client:
        page = client.get("/admin/ai-check", auth=("admin", "secret"))
    assert page.status_code == 200 and "Failing" in page.text and "API key not valid" in page.text
