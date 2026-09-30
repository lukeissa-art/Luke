import json
import sqlite3
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app import calls, db, receptionist, site
from app.main import app
from tests.conftest import NOW, FakeClaude, make_shop, reply, text

AUTH = ("admin", "secret")


@pytest.fixture
def client(settings):
    site._demo_hits.clear()
    with TestClient(app) as c:
        yield c


def test_every_shop_gets_a_widget_key(conn):
    shop = make_shop(conn)
    assert shop["widget_key"] and len(shop["widget_key"]) >= 10
    assert db.shop_by_widget_key(conn, shop["widget_key"])["id"] == shop["id"]


def test_old_databases_get_widget_keys(tmp_path):
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE shops (id INTEGER PRIMARY KEY, name TEXT, owner_phone TEXT, created_at TEXT)")
    old.execute("INSERT INTO shops (name, owner_phone, created_at) VALUES ('Old Shop', '+1', 'x')")
    old.commit()
    old.close()
    db.init_db(path)
    c = db.connect(path)
    assert c.execute("SELECT widget_key FROM shops").fetchone()[0]


def test_widget_script_and_chat_page(client, conn):
    shop = make_shop(conn)
    js = client.get("/widget.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert "data-shop" in js.text and "/chat/" in js.text and "tradedesk-chat-close" in js.text
    page = client.get(f"/chat/{shop['widget_key']}?embed=1").text
    assert "Riverside Plumbing" in page and "virtual assistant" in page and 'id="close"' in page
    assert client.get("/chat/not-a-real-key").status_code == 404


def test_paused_shop_chat_is_offline(client, conn):
    shop = make_shop(conn, status="paused")
    assert "offline" in client.get(f"/chat/{shop['widget_key']}").text
    r = client.post(f"/chat/{shop['widget_key']}/say", json={"text": "hi"})
    assert r.json()["action"] == "hangup" and "offline" in r.json()["say"]


def test_website_chat_talks_like_a_chat(client, conn, monkeypatch):
    shop = make_shop(conn)
    fake = FakeClaude([reply(text("Sorry to hear that! What's the best number to reach you?"))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake)
    r = client.post(f"/chat/{shop['widget_key']}/say", json={"text": "my sink is clogged"}).json()
    assert r["action"] == "continue" and "number" in r["say"]
    call = db.get(conn, "calls", r["chat_id"])
    assert call["call_sid"].startswith("web-") and call["caller_phone"] == ""
    req = fake.requests[0]
    assert "website chat assistant" in req["system"] and "text-to-speech" not in req["system"]
    first = req["messages"][0]["content"]
    assert first.startswith("[Website chat started") and "phone number is unknown" in first
    # a chat id from another shop is not reused
    other = make_shop(conn, name="Other", twilio_number="+15125550111")
    fake2 = FakeClaude([reply(text("Hi!"))])
    monkeypatch.setattr(receptionist, "_client", lambda: fake2)
    r2 = client.post(f"/chat/{other['widget_key']}/say", json={"text": "hi", "chat_id": r["chat_id"]}).json()
    assert r2["chat_id"] != r["chat_id"]


def test_website_chat_emergency_gives_the_shop_number(client, conn):
    shop = make_shop(conn)
    r = client.post(f"/chat/{shop['widget_key']}/say", json={"text": "I smell gas in the kitchen"}).json()
    assert r["action"] == "transfer"
    assert "(512) 555-0199" in r["say"] and "leave" in r["say"].lower()
    owner = conn.execute("SELECT body FROM sms_log WHERE to_phone = ?", (shop["owner_phone"],)).fetchall()
    assert any(row["body"].startswith("URGENT website chat") for row in owner)


def test_quiet_website_chats_are_wrapped_up(conn):
    shop = make_shop(conn)
    empty = calls.start_call(conn, shop, "web-empty", "", now=NOW)
    lead = calls.start_call(conn, shop, "web-lead", "", now=NOW)
    db.update(conn, "calls", lead["id"], {"caller_name": "Ann Lee", "callback_phone": "+15125550142",
                                          "problem": "Leaky faucet", "outcome": "message_taken"})
    assert calls.close_stale_web_chats(conn, NOW + timedelta(minutes=10)) == 0
    assert calls.close_stale_web_chats(conn, NOW + timedelta(hours=1)) == 2
    bodies = [r["body"] for r in conn.execute("SELECT body FROM sms_log WHERE to_phone = ?", (shop["owner_phone"],))]
    assert len(bodies) == 1 and bodies[0].startswith("[WEBSITE CHAT] CALL BACK: Ann Lee")
    f = conn.execute("SELECT * FROM followups").fetchone()
    assert f["to_phone"] == "+15125550142" and f["kind"] == "unbooked_caller"
    assert db.get(conn, "calls", empty["id"])["owner_notified"] == 1


def test_dashboard_shows_the_snippet(client, conn):
    shop = make_shop(conn)
    page = client.get(f"/admin/shops/{shop['id']}", auth=AUTH).text
    # the snippet sits HTML-escaped inside a read-only input, ready to copy
    assert "Website chat" in page and f"data-shop=&#34;{shop['widget_key']}&#34;" in page and "/widget.js" in page
    assert f"/chat/{shop['widget_key']}" in page


def test_phone_prompt_unchanged(conn):
    shop = make_shop(conn)
    prompt = receptionist.build_system_prompt(shop)
    assert "phone receptionist" in prompt and "text-to-speech" in prompt
    assert json.dumps(prompt)  # still a plain string


def test_chat_color_only_accepts_hex(client, conn):
    shop = make_shop(conn)
    ok = client.get(f"/chat/{shop['widget_key']}?embed=1&color=%230f766e").text
    assert "--accent: #0f766e" in ok
    bad = client.get(f"/chat/{shop['widget_key']}", params={"embed": 1, "color": "red;}body{display:none"}).text
    assert "display:none" not in bad and "--accent: red" not in bad
