import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from anthropic.types.beta import BetaTextBlock, BetaToolUseBlock

from app import db
from app.config import get_settings

# Wednesday 2026-09-30 10:00 America/Chicago (15:00 UTC)
NOW = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("RUN_WORKER", "false")
    get_settings.cache_clear()
    db.init_db()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def conn(settings):
    c = db.connect()
    yield c
    c.close()


def make_shop(conn, **overrides):
    values = {
        "name": "Riverside Plumbing",
        "trade": "plumbing",
        "owner_name": "Dave",
        "owner_phone": "+15125550100",
        "twilio_number": "+15125550199",
        "timezone": "America/Chicago",
        "business_hours": json.dumps(db.DEFAULT_HOURS),
        "slot_minutes": 120,
        "google_review_url": "https://g.page/r/review",
        "plan": "pro",
        "status": "trial",
        "created_at": db.iso(NOW),
    }
    values.update(overrides)
    shop_id = db.insert(conn, "shops", values)
    conn.commit()
    return db.get(conn, "shops", shop_id)


@pytest.fixture
def shop(conn):
    return make_shop(conn)


def text(t):
    return BetaTextBlock(type="text", text=t)


def tool(name, args, id_="tu_1"):
    return BetaToolUseBlock(type="tool_use", id=id_, name=name, input=args)


def reply(*blocks, stop=None):
    stop = stop or ("tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn")
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


class FakeClaude:
    """Returns scripted responses in order and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        if not self.responses:
            raise AssertionError("Unexpected extra Claude call")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
