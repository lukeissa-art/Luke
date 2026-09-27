import pytest
from fastapi.testclient import TestClient

from app import db, site
from app.main import app
from tests.conftest import FakeClaude, make_shop, reply, text

AUTH = ("admin", "secret")
LEAD = {"name": "Dave Ortiz", "shop_name": "Ortiz Plumbing", "phone": "(512) 555-0177", "email": "dave@example.com",
        "trade": "Plumbing & HVAC", "city": "Round Rock", "calls_per_week": "50–100", "message": "Saturdays are bad"}


@pytest.fixture
def client(settings):
    site._demo_hits.clear()
    with TestClient(app) as c:
        yield c


def test_public_pages_render(client):
    home = client.get("/")
    assert home.status_code == 200
    assert "Someone answers." in home.text and "$449" in home.text and 'action="/trial"' in home.text
    for path in ("/privacy", "/terms", "/thanks"):
        assert client.get(path).status_code == 200
    assert "do not sell or share your phone number" in client.get("/privacy").text


def test_trial_signup_json_saves_lead_and_shows_in_dashboard(client, conn):
    r = client.post("/trial", data=LEAD, headers={"Accept": "application/json"})
    assert r.json() == {"ok": True}
    lead = conn.execute("SELECT * FROM leads").fetchone()
    assert lead["phone"] == "+15125550177" and lead["status"] == "new"
    assert "Ortiz Plumbing" in client.get("/admin/", auth=AUTH).text

    form = client.get(f"/admin/shops/new?lead={lead['id']}", auth=AUTH).text
    assert 'value="Ortiz Plumbing"' in form and 'value="+15125550177"' in form
    r = client.post("/admin/shops", auth=AUTH, follow_redirects=False, data={
        "lead_id": str(lead["id"]), "name": "Ortiz Plumbing", "trade": "hvac", "owner_phone": "+15125550177",
        "plan": "pro", "answer_mode": "after_hours", "timezone": "America/Chicago"})
    assert r.status_code == 303
    lead = db.get(conn, "leads", lead["id"])
    assert lead["status"] == "converted" and lead["shop_id"]


def test_trial_signup_without_js_redirects(client, conn):
    r = client.post("/trial", data=LEAD, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/thanks"


def test_trial_signup_validation(client, conn):
    r = client.post("/trial", data={**LEAD, "phone": "123", "name": ""}, headers={"Accept": "application/json"})
    assert r.status_code == 422 and set(r.json()["errors"]) == {"phone", "name"}
    r = client.post("/trial", data={**LEAD, "phone": "123"})
    assert r.status_code == 422 and "10-digit" in r.text
    assert conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0] == 0


def test_honeypot_drops_bots(client, conn):
    r = client.post("/trial", data={**LEAD, "website": "http://spam"}, headers={"Accept": "application/json"})
    assert r.json() == {"ok": True}
    assert conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0] == 0


def test_public_demo_off_by_default(client):
    assert client.post("/demo/say", json={"text": "hi"}).status_code == 404


def test_public_demo_talks_and_texts_nobody(client, conn, monkeypatch):
    from app import receptionist

    shop = make_shop(conn)
    monkeypatch.setenv("PUBLIC_DEMO_SHOP_ID", str(shop["id"]))
    from app.config import get_settings
    get_settings.cache_clear()
    fake = FakeClaude([reply(text("Sorry to hear that. What's the address?"))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)

    d = client.post("/demo/say", json={"text": "my sink is clogged"}).json()
    assert d["action"] == "continue" and "address" in d["say"]
    assert db.get(conn, "calls", d["call_id"])["call_sid"].startswith("demo-web-")

    # gas smell: transferred by the fixed rules, and the demo shop owner is not texted
    d = client.post("/demo/say", json={"text": "I smell gas", "call_id": d["call_id"]}).json()
    assert d["action"] == "transfer"
    assert conn.execute("SELECT COUNT(*) FROM sms_log").fetchone()[0] == 0


def test_public_demo_rate_limit(client, conn, monkeypatch):
    shop = make_shop(conn)
    monkeypatch.setenv("PUBLIC_DEMO_SHOP_ID", str(shop["id"]))
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setattr(site, "DEMO_MAX_MESSAGES_PER_HOUR", 2)
    for _ in range(2):
        client.post("/demo/say", json={"text": "I smell gas"})
    assert client.post("/demo/say", json={"text": "hello"}).status_code == 429


def test_health_answers_while_the_ai_is_thinking(settings, conn, monkeypatch):
    """A slow AI reply must not freeze the server: Render restarts it if /health stalls."""
    import threading
    import time as _time

    import httpx
    import uvicorn

    from app import receptionist

    shop = make_shop(conn)
    monkeypatch.setenv("PUBLIC_DEMO_SHOP_ID", str(shop["id"]))
    from app.config import get_settings
    get_settings.cache_clear()

    class SlowBackend:
        def __init__(self, *a, **k):
            pass

        def user_message(self, t):
            return {"role": "user", "content": t}

        def assistant_message(self, t):
            return {"role": "assistant", "content": t}

        def step(self, *a):
            _time.sleep(3)
            from app.llm import Step
            return Step(spoken=["Got it."], assistant_message={"role": "assistant", "content": "Got it."})

    monkeypatch.setattr(receptionist, "make_backend", lambda client=None: SlowBackend())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8799, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        _time.sleep(0.05)
    try:
        slow = threading.Thread(target=lambda: httpx.post("http://127.0.0.1:8799/demo/say",
                                                          json={"text": "hi"}, timeout=10))
        slow.start()
        _time.sleep(0.5)
        began = _time.monotonic()
        assert httpx.get("http://127.0.0.1:8799/health", timeout=5).status_code == 200
        assert _time.monotonic() - began < 1.0
        slow.join()
    finally:
        server.should_exit = True
        t.join(timeout=5)
