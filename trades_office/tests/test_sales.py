import pytest
from fastapi.testclient import TestClient

from app import llm, site
from app.main import app
from app.receptionist import build_system_prompt
from tests.conftest import make_shop


class RecordingBackend:
    def __init__(self, reply="Pro is $449 a month.", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def user_message(self, t):
        return {"role": "user", "content": t}

    def assistant_message(self, t):
        return {"role": "assistant", "content": t}

    def step(self, system, tools, messages):
        self.calls.append((system, tools, messages))
        if self.error:
            raise self.error
        return llm.Step(spoken=[self.reply], assistant_message=self.assistant_message(self.reply))


@pytest.fixture
def client(settings):
    site._demo_hits.clear()
    with TestClient(app) as c:
        yield c


def test_ask_answers_with_company_facts(client, monkeypatch):
    backend = RecordingBackend()
    monkeypatch.setattr("app.sales.make_backend", lambda: backend)
    r = client.post("/ask", json={"question": "How much is it?",
                                  "history": [{"role": "assistant", "text": "Hi!"},
                                              {"role": "user", "text": "hello"},
                                              {"role": "assistant", "text": "Hi, ask away."}]})
    assert r.json() == {"answer": "Pro is $449 a month."}
    system, tools, messages = backend.calls[0]
    assert tools == []
    for fact in ("$249/month", "$449/month", "$799/month", "$500", "14 days", "no card"):
        assert fact in system
    # history starts with the visitor, ends with the new question
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[-1] == {"role": "user", "content": "How much is it?"}


def test_ask_validation_and_failure(client, monkeypatch):
    assert client.post("/ask", json={"question": "  "}).status_code == 400
    monkeypatch.setattr("app.sales.make_backend", lambda: RecordingBackend(error=RuntimeError("boom")))
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 200 and "email" in r.json()["answer"]


def test_home_page_has_no_floating_chat_button(client):
    # The owner preferred the page without the floating "Questions? Ask us" button.
    page = client.get("/").text
    assert 'id="ask-panel"' not in page and "Questions? Ask us" not in page


def test_receptionist_prompt_has_shop_facts(conn):
    shop = make_shop(conn, pricing_notes="$89 service call, waived with repair.",
                     service_area="Austin", services="Drains, water heaters")
    prompt = build_system_prompt(shop)
    assert "$89 service call, waived with repair." in prompt
    assert "Monday 8 AM to 5 PM" in prompt and "closed Saturday, Sunday" in prompt
    assert "2-hour arrival windows" in prompt
    assert "don't guess" in prompt


def test_gemini_omits_tools_when_there_are_none():
    from types import SimpleNamespace

    from google.genai import types

    seen = {}

    def generate(*, model, contents, config):
        seen["tools"] = config.tools
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text="Hi")]), finish_reason="STOP")])

    backend = llm.GeminiBackend(SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    step = backend.step("sys", [], [backend.user_message("hi")])
    assert step.spoken == ["Hi"] and seen["tools"] is None


def test_assistant_knows_the_whole_website(settings):
    from app import sales

    sales.website_text.cache_clear()
    text = sales.website_text()
    # home page: hero, emergency rules, FAQ answers, comparison
    for snippet in ("The phone rings at 9:47 PM", "Emergencies aren't left to the AI",
                    "Do I need a new phone number?", "You keep your number", "Answering service"):
        assert snippet in text, snippet
    # privacy policy and terms
    assert "Reply STOP to opt out" in text and "Terms of service" in text
    # scripts and chat widgets aren't included
    assert "fetch(" not in text and "Ask about" not in text
    starter = text[text.index("## Starter"):text.index("## Pro")]
    assert "Not included: Books into your calendar" in starter
    assert text in sales.system_prompt()
