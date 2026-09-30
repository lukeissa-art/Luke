"""Founder dashboard: onboard shops, watch calls, mark jobs done, run live demos."""

import json
import secrets
import uuid
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from . import calls, db, followups, phone_numbers, reports, scheduling
from .config import get_settings
from .plans import PLANS, TRIAL_DAYS, get_plan
from .receptionist import Receptionist, greeting, normalize_phone, pretty_phone
from .widget import chat_link as widget_link, snippet as widget_snippet

security = HTTPBasic()


def require_admin(creds: HTTPBasicCredentials = Depends(security)) -> str:
    s = get_settings()
    if s.admin_password == "change-me" and not s.is_local:
        raise HTTPException(status_code=503, detail="Set ADMIN_PASSWORD before using the dashboard.")
    ok = secrets.compare_digest(creds.username, s.admin_username) and secrets.compare_digest(
        creds.password, s.admin_password
    )
    if not ok:
        raise HTTPException(status_code=401, headers={"WWW-Authenticate": "Basic"})
    return creds.username


router = APIRouter(dependencies=[Depends(require_admin)])
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["phone"] = pretty_phone
templates.env.globals["company"] = lambda: get_settings().company_name

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
SHOP_FIELDS = ["name", "trade", "owner_name", "owner_phone", "twilio_number", "timezone", "service_area",
               "services", "pricing_notes", "extra_instructions", "emergency_keywords", "calendar_provider",
               "google_calendar_id", "google_review_url", "answer_mode", "plan"]


def _render(request: Request, name: str, **ctx):
    return templates.TemplateResponse(request, name, ctx)


@router.get("/", response_class=HTMLResponse)
def overview(request: Request):
    now = db.utcnow()
    with db.session() as conn:
        shops = conn.execute("SELECT * FROM shops ORDER BY created_at DESC").fetchall()
        leads = conn.execute("SELECT * FROM leads WHERE status IN ('new', 'contacted') "
                             "ORDER BY created_at DESC LIMIT 50").fetchall()
        rows = []
        for shop in shops:
            st = reports.stats(conn, shop, now - timedelta(days=7), now)
            rows.append({"shop": shop, "stats": st, "plan": get_plan(shop["plan"])})
    mrr = sum(r["plan"].monthly_price for r in rows if r["shop"]["status"] == "active")
    trials = sum(1 for r in rows if r["shop"]["status"] == "trial")
    from . import llm

    return _render(request, "index.html", rows=rows, leads=leads, ai_error=dict(llm.last_error), mrr=mrr, arr=mrr * 12, trials=trials,
                   active=sum(1 for r in rows if r["shop"]["status"] == "active"),
                   goal_shops=60)


TRADES = {"plumbing": "plumbing", "hvac": "hvac", "plumbing & hvac": "hvac", "electrical": "electrical"}


@router.get("/shops/new", response_class=HTMLResponse)
def new_shop(request: Request, lead: int = 0):
    prefill = None
    if lead:
        with db.session() as conn:
            row = db.get(conn, "leads", lead)
        if row:
            prefill = {"lead_id": row["id"], "name": row["shop_name"], "owner_name": row["name"],
                       "owner_phone": row["phone"], "trade": TRADES.get(row["trade"].lower(), "plumbing"),
                       "service_area": row["city"], "extra_instructions": row["message"]}
    return _render(request, "shop_form.html", shop=None, prefill=prefill, hours=db.DEFAULT_HOURS,
                   plans=PLANS, days=DAYS)


@router.post("/leads/{lead_id}/{status}")
def set_lead_status(lead_id: int, status: str):
    if status not in ("contacted", "lost"):
        _404()
    with db.session() as conn:
        db.get(conn, "leads", lead_id) or _404()
        db.update(conn, "leads", lead_id, {"status": status})
    return RedirectResponse("/admin/", status_code=303)


def _shop_values(form) -> dict:
    values = {k: (form.get(k) or "").strip() for k in SHOP_FIELDS}
    values["owner_phone"] = normalize_phone(values["owner_phone"]) or values["owner_phone"]
    values["twilio_number"] = normalize_phone(values["twilio_number"]) or None
    hours = {}
    for d in DAYS:
        open_t, close_t = form.get(f"{d}_open"), form.get(f"{d}_close")
        if open_t and close_t:
            hours[d] = [open_t, close_t]
    values["business_hours"] = json.dumps(hours)
    values["slot_minutes"] = int(form.get("slot_minutes") or 120)
    values["avg_job_value"] = int(form.get("avg_job_value") or 450)
    if values["plan"] == "starter":
        values["answer_mode"] = "after_hours"
    return values


@router.post("/shops")
async def create_shop(request: Request):
    form = await request.form()
    values = _shop_values(form)
    now = db.utcnow()
    values.update(status="trial", trial_ends_at=db.iso(now + timedelta(days=TRIAL_DAYS)), created_at=db.iso(now))
    with db.session() as conn:
        shop_id = db.insert(conn, "shops", values)
        if str(form.get("lead_id") or "").isdigit():
            db.update(conn, "leads", int(form["lead_id"]), {"status": "converted", "shop_id": shop_id})
    return RedirectResponse(f"/admin/shops/{shop_id}", status_code=303)


@router.get("/shops/{shop_id}", response_class=HTMLResponse)
def shop_detail(request: Request, shop_id: int, phone_ok: str = "", phone_err: str = ""):
    now = db.utcnow()
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id) or _404()
        ctx = dict(
            shop=shop,
            plan=get_plan(shop["plan"]),
            stats=reports.stats(conn, shop, now - timedelta(days=7), now),
            month_calls=calls.calls_this_month(conn, shop_id),
            calls=conn.execute("SELECT * FROM calls WHERE shop_id = ? ORDER BY started_at DESC LIMIT 25",
                               (shop_id,)).fetchall(),
            appts=conn.execute("SELECT * FROM appointments WHERE shop_id = ? AND status = 'booked' "
                               "ORDER BY start_at LIMIT 25", (shop_id,)).fetchall(),
            followups=conn.execute("SELECT * FROM followups WHERE shop_id = ? ORDER BY send_at DESC LIMIT 25",
                                   (shop_id,)).fetchall(),
            quotes=conn.execute("SELECT * FROM quotes WHERE shop_id = ? ORDER BY created_at DESC LIMIT 25",
                                (shop_id,)).fetchall(),
            slots=scheduling.open_slots(conn, shop, limit=3),
            label=lambda iso: scheduling.label(shop, iso),
            base_url=get_settings().public_base_url,
            widget_snippet=widget_snippet(shop["widget_key"]) if shop["widget_key"] else "",
            widget_link=widget_link(shop["widget_key"]) if shop["widget_key"] else "",
            twilio_enabled=get_settings().twilio_enabled,
            phone_ok=phone_ok[:300],
            phone_err=phone_err[:300],
        )
    return _render(request, "shop.html", **ctx)


@router.get("/shops/{shop_id}/edit", response_class=HTMLResponse)
def edit_shop(request: Request, shop_id: int):
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id) or _404()
    return _render(request, "shop_form.html", shop=shop, hours=db.business_hours(shop), plans=PLANS, days=DAYS)


@router.post("/shops/{shop_id}")
async def update_shop(request: Request, shop_id: int):
    form = await request.form()
    values = _shop_values(form)
    if form.get("status"):
        values["status"] = form["status"]
    with db.session() as conn:
        db.update(conn, "shops", shop_id, values)
    return RedirectResponse(f"/admin/shops/{shop_id}", status_code=303)


def _back_to_shop(shop_id: int, **msg) -> RedirectResponse:
    return RedirectResponse(f"/admin/shops/{shop_id}?{urlencode(msg)}#phone", status_code=303)


def _number_taken(conn, number: str, shop_id: int) -> bool:
    other = db.shop_by_number(conn, number)
    return other is not None and other["id"] != shop_id


@router.post("/shops/{shop_id}/phone/buy")
def buy_phone_number(shop_id: int, area_code: str = Form("")):
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id) or _404()
    if shop["twilio_number"]:
        return _back_to_shop(shop_id, phone_err="This shop already has an assistant number. Release it first.")
    try:
        number = phone_numbers.buy_number(area_code, shop["name"])
    except phone_numbers.PhoneNumberError as exc:
        return _back_to_shop(shop_id, phone_err=str(exc))
    with db.session() as conn:
        db.update(conn, "shops", shop_id, {"twilio_number": number})
    return _back_to_shop(shop_id, phone_ok=f"Done: {pretty_phone(number)} is now this shop's assistant line "
                                           "and is ready to take calls and texts.")


@router.post("/shops/{shop_id}/phone/connect")
def connect_phone_number(shop_id: int, number: str = Form("")):
    e164 = normalize_phone(number)
    if not e164:
        return _back_to_shop(shop_id, phone_err="Enter a 10-digit US phone number.")
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id) or _404()
        if _number_taken(conn, e164, shop_id):
            return _back_to_shop(shop_id, phone_err=f"{pretty_phone(e164)} is already used by another shop.")
    try:
        e164 = phone_numbers.connect_number(e164, shop["name"])
    except phone_numbers.PhoneNumberError as exc:
        return _back_to_shop(shop_id, phone_err=str(exc))
    with db.session() as conn:
        db.update(conn, "shops", shop_id, {"twilio_number": e164})
    return _back_to_shop(shop_id, phone_ok=f"Connected: calls and texts to {pretty_phone(e164)} now go to "
                                           "this shop's assistant.")


@router.post("/shops/{shop_id}/phone/release")
def release_phone_number(shop_id: int):
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id) or _404()
    if not shop["twilio_number"]:
        return _back_to_shop(shop_id)
    try:
        phone_numbers.release_number(shop["twilio_number"])
    except phone_numbers.PhoneNumberError as exc:
        return _back_to_shop(shop_id, phone_err=str(exc))
    with db.session() as conn:
        db.update(conn, "shops", shop_id, {"twilio_number": None})
    return _back_to_shop(shop_id, phone_ok=f"Released {pretty_phone(shop['twilio_number'])}. "
                                           "It no longer rings the assistant or costs anything.")


@router.get("/calls/{call_id}", response_class=HTMLResponse)
def call_detail(request: Request, call_id: int):
    with db.session() as conn:
        call = db.get(conn, "calls", call_id) or _404()
        shop = db.get(conn, "shops", call["shop_id"])
    return _render(request, "call.html", call=call, shop=shop, transcript=json.loads(call["transcript"]))


@router.post("/appointments/{appt_id}/done")
def appointment_done(appt_id: int):
    with db.session() as conn:
        appt = db.get(conn, "appointments", appt_id) or _404()
        calls.complete_appointment(conn, appt_id)
    return RedirectResponse(f"/admin/shops/{appt['shop_id']}", status_code=303)


@router.post("/shops/{shop_id}/quotes")
def add_quote(shop_id: int, customer_name: str = Form(""), customer_phone: str = Form(...),
              amount: float = Form(0), description: str = Form("")):
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id) or _404()
        qid = db.insert(conn, "quotes", {
            "shop_id": shop_id, "customer_name": customer_name,
            "customer_phone": normalize_phone(customer_phone) or customer_phone,
            "amount": amount, "description": description, "created_at": db.iso(db.utcnow()),
        })
        followups.schedule_quote_followups(conn, shop, db.get(conn, "quotes", qid), now=db.utcnow())
    return RedirectResponse(f"/admin/shops/{shop_id}", status_code=303)


@router.post("/quotes/{quote_id}/{status}")
def set_quote_status(quote_id: int, status: str):
    if status not in ("won", "lost"):
        _404()
    with db.session() as conn:
        quote = db.get(conn, "quotes", quote_id) or _404()
        db.update(conn, "quotes", quote_id, {"status": status})
    return RedirectResponse(f"/admin/shops/{quote['shop_id']}", status_code=303)


@router.post("/shops/{shop_id}/checkout")
def create_checkout(shop_id: int, waive_setup_fee: bool = Form(False)):
    from .billing import checkout_url

    if not get_settings().stripe_api_key:
        raise HTTPException(status_code=400, detail="Set STRIPE_API_KEY and price IDs first")
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id) or _404()
    return RedirectResponse(checkout_url(shop, waive_setup_fee=waive_setup_fee), status_code=303)


# --- Live demo: talk to a shop's assistant in the browser (sales visits, setup test calls) ---


@router.get("/shops/{shop_id}/demo", response_class=HTMLResponse)
def demo(request: Request, shop_id: int):
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id) or _404()
    return _render(request, "demo.html", shop=shop, greeting=greeting(shop))


@router.post("/shops/{shop_id}/demo/say")
async def demo_say(shop_id: int, request: Request):
    body = await request.json()
    return await run_in_threadpool(_demo_say, shop_id, body)  # AI call: keep the event loop free


def _demo_say(shop_id: int, body: dict):
    with db.session() as conn:
        shop = db.get(conn, "shops", shop_id) or _404()
        call = db.get(conn, "calls", int(body["call_id"])) if body.get("call_id") else None
        if call is None or call["shop_id"] != shop_id:
            call = calls.start_call(conn, shop, f"demo-{uuid.uuid4().hex[:12]}", "+15555550100")
        if call["owner_notified"]:
            return JSONResponse({"say": "(this test call has ended)", "action": "hangup", "call_id": call["id"]})
        turn = Receptionist(conn, shop, call).respond(body.get("text", ""))
        if turn.action != "continue":
            calls.finalize_call(conn, call["id"],
                                status="transferred" if turn.action == "transfer" else "completed")
    return JSONResponse({"say": turn.say, "action": turn.action, "call_id": call["id"]})


@router.get("/ai-check", response_class=HTMLResponse)
def ai_check(request: Request):
    """Run one tiny request against the configured AI model and show exactly what happened."""
    import time

    from . import llm
    from .receptionist import TOOL_END, make_backend

    s = get_settings()
    prov = llm.provider()
    key = s.gemini_api_key if prov == "gemini" else s.anthropic_api_key
    info = {"provider": prov, "model": llm.current_model(),
            "key": f"set (ends in …{key[-4:]})" if len(key) > 8 else "MISSING"}
    started = time.monotonic()
    try:
        backend = make_backend()
        step = backend.step(
            "You are testing a phone assistant. Reply with a short friendly sentence.",
            [TOOL_END],
            [backend.user_message("Hi, is anyone there?")],
        )
        ok = not step.refused
        result = " ".join(step.spoken) or ("(model declined)" if step.refused else "(tool call only)")
    except Exception as exc:
        ok, result = False, f"{type(exc).__name__}: {exc}"
    info["model_used"] = llm.current_model()
    info["seconds"] = f"{time.monotonic() - started:.1f}"
    return _render(request, "ai_check.html", ok=ok, result=result, info=info)


def _404():
    raise HTTPException(status_code=404)
