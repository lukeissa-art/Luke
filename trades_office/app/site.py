"""Public website: landing page, trial signup, live demo, privacy and terms."""

import logging
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import calls, db, sms
from .config import get_settings
from .plans import PLANS, SETUP_FEE, TRIAL_DAYS
from .receptionist import Receptionist, greeting, normalize_phone

log = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

DEMO_MAX_MESSAGES_PER_HOUR = 20  # per visitor IP; each message is one Claude call
DEMO_MAX_TURNS_PER_CALL = 16
DEMO_MAX_CHARS = 300
_demo_hits: dict[str, deque] = defaultdict(deque)


def site_context(**extra) -> dict:
    s = get_settings()
    return {
        "company": s.company_name,
        "contact_email": s.contact_email,
        "demo_phone": s.demo_phone_number,
        "region": s.service_region,
        "plans": PLANS,
        "setup_fee": SETUP_FEE,
        "trial_days": TRIAL_DAYS,
        "live_demo": bool(s.public_demo_shop_id),
        "fragment": False,
        **extra,
    }


@router.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse(request, "site/landing.html", site_context())


@router.get("/privacy", response_class=HTMLResponse)
def privacy(request: Request):
    return templates.TemplateResponse(request, "site/privacy.html", site_context())


@router.get("/terms", response_class=HTMLResponse)
def terms(request: Request):
    return templates.TemplateResponse(request, "site/terms.html", site_context())


@router.get("/thanks", response_class=HTMLResponse)
def thanks(request: Request):
    return templates.TemplateResponse(request, "site/thanks.html", site_context())


def _validate_lead(form) -> tuple[dict, dict]:
    values = {k: (str(form.get(k) or "")).strip()[:500] for k in
              ("name", "shop_name", "phone", "email", "trade", "city", "calls_per_week", "message")}
    errors = {}
    if not values["name"]:
        errors["name"] = "Add your name so we know who to ask for."
    if not values["shop_name"]:
        errors["shop_name"] = "Add your shop's name."
    phone = normalize_phone(values["phone"])
    if not phone:
        errors["phone"] = "Enter a 10-digit US phone number, like 512-555-0123."
    values["phone"] = phone or values["phone"]
    if values["email"] and "@" not in values["email"]:
        errors["email"] = "That email address is missing an @."
    return values, errors


@router.post("/trial")
async def trial_signup(request: Request):
    form = await request.form()
    wants_json = "application/json" in request.headers.get("accept", "")
    if form.get("website"):  # honeypot field, hidden from people; bots fill it in
        return JSONResponse({"ok": True}) if wants_json else RedirectResponse("/thanks", status_code=303)
    values, errors = _validate_lead(form)
    if errors:
        if wants_json:
            return JSONResponse({"ok": False, "errors": errors}, status_code=422)
        return templates.TemplateResponse(request, "site/landing.html",
                                          site_context(errors=errors, form=values), status_code=422)
    with db.session() as conn:
        lead_id = db.insert(conn, "leads", {**values, "created_at": db.iso(db.utcnow())})
        sms.alert_founder(conn, f"New trial signup #{lead_id}: {values['name']}, {values['shop_name']} "
                                f"({values['trade'] or 'trade?'}, {values['city'] or 'city?'}) {values['phone']}. "
                                f"Calls/week: {values['calls_per_week'] or '?'}")
    return JSONResponse({"ok": True}) if wants_json else RedirectResponse("/thanks", status_code=303)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def _rate_limited(ip: str) -> bool:
    now = time.monotonic()
    hits = _demo_hits[ip]
    while hits and now - hits[0] > 3600:
        hits.popleft()
    if len(hits) >= DEMO_MAX_MESSAGES_PER_HOUR:
        return True
    hits.append(now)
    return False


@router.post("/demo/say")
async def public_demo(request: Request):
    shop_id = get_settings().public_demo_shop_id
    if not shop_id:
        return JSONResponse({"error": "The live demo is turned off."}, status_code=404)
    if _rate_limited(_client_ip(request)):
        return JSONResponse({"error": "You've reached the demo limit for now. Start a free trial to keep going."},
                            status_code=429)
    body = await request.json()
    text = str(body.get("text", ""))[:DEMO_MAX_CHARS]
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id)
        if shop is None:
            return JSONResponse({"error": "The live demo is turned off."}, status_code=404)
        call = None
        if body.get("call_id"):
            call = db.get(conn, "calls", int(body["call_id"]))
            if call is None or call["shop_id"] != shop_id or not call["call_sid"].startswith("demo-web-"):
                call = None
        if call is None:
            call = calls.start_call(conn, shop, f"demo-web-{uuid.uuid4().hex[:12]}", "+15555550100")
        if call["owner_notified"] or call["transcript"].count('"caller"') >= DEMO_MAX_TURNS_PER_CALL:
            calls.finalize_call(conn, call["id"])
            return JSONResponse({"say": "That's the end of this demo call. Start a new one anytime.",
                                 "action": "hangup", "call_id": call["id"]})
        turn = Receptionist(conn, shop, call).respond(text)
        if turn.action != "continue":
            calls.finalize_call(conn, call["id"])
    return JSONResponse({"say": turn.say, "action": turn.action, "call_id": call["id"]})


@router.get("/demo/greeting")
def public_demo_greeting():
    shop_id = get_settings().public_demo_shop_id
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id) if shop_id else None
    if shop is None:
        return JSONResponse({"error": "The live demo is turned off."}, status_code=404)
    return {"greeting": greeting(shop), "shop": shop["name"]}
