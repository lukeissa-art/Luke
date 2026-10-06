from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import db, sms
from app.config import get_settings
from app.main import app
from tests.conftest import make_shop

AUTH = ("admin", "secret")


class FakeTwilio:
    """Just enough of twilio.rest.Client for buying, connecting and releasing numbers."""

    def __init__(self, available=("+15125550142",), owned=()):
        self.available = list(available)
        self.owned = {n: SimpleNamespace(sid=f"PN{i}", phone_number=n) for i, n in enumerate(owned)}
        self.searches, self.created, self.updated, self.deleted = [], [], [], []
        fake = self

        class Local:
            def list(self, **kw):
                fake.searches.append(kw)
                return [SimpleNamespace(phone_number=n) for n in fake.available[: kw.get("limit", 20)]]

        class Incoming:
            def __call__(self, sid):
                return SimpleNamespace(update=lambda **kw: fake.updated.append((sid, kw)),
                                       delete=lambda: fake.deleted.append(sid))

            def create(self, **kw):
                fake.created.append(kw)
                num = SimpleNamespace(sid="PNnew", phone_number=kw["phone_number"])
                fake.owned[num.phone_number] = num
                return num

            def list(self, phone_number=None, limit=None):
                return [fake.owned[phone_number]] if phone_number in fake.owned else []

        self._local = Local()
        self.incoming_phone_numbers = Incoming()

    def available_phone_numbers(self, country):
        assert country == "US"
        return SimpleNamespace(local=self._local)


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
    with TestClient(app) as c:
        yield c


def _post(client, path, **data):
    return client.post(path, data=data, auth=AUTH, follow_redirects=False)


def test_get_a_phone_number_buys_connects_and_saves(client, conn, twilio_on):
    shop = make_shop(conn, twilio_number=None)
    r = _post(client, f"/admin/shops/{shop['id']}/phone/buy", area_code="512")
    assert r.status_code == 303 and "phone_ok" in r.headers["location"]
    assert twilio_on.searches[0]["area_code"] == "512"
    assert twilio_on.searches[0]["voice_enabled"] and twilio_on.searches[0]["sms_enabled"]
    created = twilio_on.created[0]
    assert created["phone_number"] == "+15125550142"
    assert created["voice_url"] == "https://tradedesk.example.com/voice/incoming"
    assert created["status_callback"] == "https://tradedesk.example.com/voice/status"
    assert created["sms_url"] == "https://tradedesk.example.com/sms/incoming"
    assert db.get(conn, "shops", shop["id"])["twilio_number"] == "+15125550142"

    page = client.get(r.headers["location"], auth=AUTH).text
    assert "(512) 555-0142" in page and "now this shop&#39;s assistant line" in page
    assert "*725125550142" in page  # Verizon forwarding code for the new number


def test_buy_reports_problems_without_saving(client, conn, twilio_on):
    shop = make_shop(conn, twilio_number=None)
    r = _post(client, f"/admin/shops/{shop['id']}/phone/buy", area_code="51")
    assert "phone_err" in r.headers["location"] and not twilio_on.created

    twilio_on.available = []
    r = _post(client, f"/admin/shops/{shop['id']}/phone/buy", area_code="907")
    page = client.get(r.headers["location"], auth=AUTH).text
    assert "No numbers are available in area code 907" in page
    assert db.get(conn, "shops", shop["id"])["twilio_number"] is None


def test_buy_refuses_when_shop_already_has_a_number(client, conn, twilio_on):
    shop = make_shop(conn)
    r = _post(client, f"/admin/shops/{shop['id']}/phone/buy", area_code="512")
    assert "phone_err" in r.headers["location"] and not twilio_on.created


def test_buttons_explain_when_twilio_is_not_connected(client, conn):
    shop = make_shop(conn, twilio_number=None)
    page = client.get(f"/admin/shops/{shop['id']}", auth=AUTH).text
    assert "Twilio isn't connected" in page
    r = _post(client, f"/admin/shops/{shop['id']}/phone/buy", area_code="512")
    page = client.get(r.headers["location"], auth=AUTH).text
    assert "TWILIO_ACCOUNT_SID" in page
    assert db.get(conn, "shops", shop["id"])["twilio_number"] is None


def test_connect_existing_number(client, conn, twilio_on):
    shop = make_shop(conn, twilio_number=None)
    r = _post(client, f"/admin/shops/{shop['id']}/phone/connect", number="(512) 555-0188")
    assert "phone_ok" in r.headers["location"]
    sid, kw = twilio_on.updated[0]
    assert sid == "PN0" and kw["voice_url"].endswith("/voice/incoming") and kw["sms_url"].endswith("/sms/incoming")
    assert db.get(conn, "shops", shop["id"])["twilio_number"] == "+15125550188"


def test_connect_rejects_unknown_or_taken_numbers(client, conn, twilio_on):
    make_shop(conn, name="Other Shop", twilio_number="+15125550188")
    shop = make_shop(conn, twilio_number=None)
    r = _post(client, f"/admin/shops/{shop['id']}/phone/connect", number="512-555-0188")
    assert "already+used" in r.headers["location"]
    r = _post(client, f"/admin/shops/{shop['id']}/phone/connect", number="512-555-0111")
    assert "isn%27t+on+your+Twilio+account" in r.headers["location"]
    assert not twilio_on.updated
    assert db.get(conn, "shops", shop["id"])["twilio_number"] is None


def test_release_number(client, conn, twilio_on):
    shop = make_shop(conn, twilio_number="+15125550188")
    r = _post(client, f"/admin/shops/{shop['id']}/phone/release")
    assert "phone_ok" in r.headers["location"]
    assert twilio_on.deleted == ["PN0"]
    assert db.get(conn, "shops", shop["id"])["twilio_number"] is None


def _fallback_twiml(url):
    from urllib.parse import parse_qs, urlparse

    assert url.startswith("https://twimlets.com/echo?")
    return parse_qs(urlparse(url).query)["Twiml"][0]


def test_new_numbers_fall_back_to_the_owner_if_the_app_is_down(client, conn, twilio_on):
    shop = make_shop(conn, twilio_number=None, name="Riverside Plumbing & Drain")
    _post(client, f"/admin/shops/{shop['id']}/phone/buy", area_code="512")
    created = twilio_on.created[0]
    assert created["voice_fallback_method"] == "GET"
    twiml = _fallback_twiml(created["voice_fallback_url"])
    assert "<Dial timeout=\"25\">+15125550100</Dial>" in twiml
    assert "Riverside Plumbing &amp; Drain" in twiml  # valid XML even with special characters
    import xml.dom.minidom
    xml.dom.minidom.parseString(twiml)

    page = client.get(f"/admin/shops/{shop['id']}", auth=AUTH).text
    assert "Backup: if this app is ever down" in page


def test_no_fallback_without_an_owner_phone(conn, twilio_on):
    from app import phone_numbers

    shop = make_shop(conn, owner_phone="")
    assert phone_numbers.fallback_url(shop) == ""
    assert "voice_fallback_url" not in phone_numbers.webhooks(shop)


def _edit_form(shop, **changes):
    form = {k: shop[k] or "" for k in ("name", "trade", "owner_name", "owner_phone", "twilio_number", "timezone",
                                        "service_area", "services", "pricing_notes", "plan")}
    form.update(mon_open="08:00", mon_close="17:00", slot_minutes="120", avg_job_value="450", **changes)
    return form


def test_changing_the_owners_phone_updates_the_fallback(client, conn, twilio_on):
    shop = make_shop(conn, twilio_number="+15125550188")
    client.post(f"/admin/shops/{shop['id']}", data=_edit_form(shop, owner_phone="(512) 555-0177"),
                auth=AUTH, follow_redirects=False)
    sid, kw = twilio_on.updated[-1]
    assert sid == "PN0" and "+15125550177" in _fallback_twiml(kw["voice_fallback_url"])

    # Saving without touching the phone or name doesn't call Twilio again
    before = len(twilio_on.updated)
    shop = db.get(conn, "shops", shop["id"])
    client.post(f"/admin/shops/{shop['id']}", data=_edit_form(shop, services="drains"), auth=AUTH)
    assert len(twilio_on.updated) == before
