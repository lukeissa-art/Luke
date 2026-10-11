"""Twilio webhooks: voice (answer, converse, transfer) and inbound SMS.

Setup per shop: buy a Twilio number, point its Voice webhook at /voice/incoming and its
Messaging webhook at /sms/incoming. The shop forwards calls to that number (all calls for
24/7 plans, or no-answer/after-hours forwarding for Starter).
"""

import logging
import time
from urllib.parse import urlencode
from xml.sax.saxutils import escape

from fastapi import APIRouter, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from . import calls, db, followups, scheduling, sms
from .config import get_settings
from .plans import get_plan
from .receptionist import Receptionist, greeting, normalize_phone

log = logging.getLogger(__name__)
router = APIRouter()


def twiml(inner: str) -> Response:
    return Response(f'<?xml version="1.0" encoding="UTF-8"?><Response>{inner}</Response>',
                    media_type="application/xml")


def say(text: str) -> str:
    return f'<Say voice="{get_settings().tts_voice}">{escape(text)}</Say>'


def gather(prompt: str) -> str:
    s = get_settings()
    action = f"{s.public_base_url}/voice/respond"
    return (
        f'<Gather input="speech" action="{action}" method="POST" speechTimeout="auto" '
        f'speechModel="phone_call" enhanced="true" actionOnEmptyResult="true" timeout="6">'
        f"{say(prompt)}</Gather>"
        f'<Redirect method="POST">{action}</Redirect>'
    )


# Call screening. When we ring the owner, they first hear "press 1 to take it". Their voicemail
# can't press 1, so a call they don't really answer comes back to the assistant instead of
# ending up in the owner's voicemail box.
_accepted: dict[str, float] = {}  # caller's CallSid -> when the owner pressed 1
_ACCEPTED_TTL = 6 * 3600


def _mark_accepted(call_sid: str) -> None:
    now = time.time()
    for sid, at in list(_accepted.items()):
        if now - at > _ACCEPTED_TTL:
            del _accepted[sid]
    _accepted[call_sid] = now


def _was_accepted(call_sid: str) -> bool:
    return _accepted.pop(call_sid, None) is not None


# Loop guard. If the owner's cell is the business line and forwards to the assistant, ringing the
# owner sends the call straight back to us. We spot the bounce and answer it "busy", so the
# original call falls through to the assistant instead of bouncing back and forth.
_ringing_owner: dict[str, tuple[str, float]] = {}  # "shop:caller" -> (caller's CallSid, started)
_BOUNCE_WINDOW = 60


def _is_bounce(shop, p: dict[str, str]) -> bool:
    caller, to = normalize_phone(p.get("From", "")), normalize_phone(p.get("To", ""))
    if caller and caller == to:
        return True  # our own transfer (caller ID = the shop's number) forwarded back to us
    if not caller:
        return False  # blocked/anonymous callers all look alike: never treat them as a bounce
    started = _ringing_owner.get(f'{shop["id"]}:{caller}')
    return bool(started and started[0] != p.get("CallSid") and time.time() - started[1] < _BOUNCE_WINDOW)


def _start_ringing(shop, p: dict[str, str]) -> None:
    _ringing_owner[f'{shop["id"]}:{normalize_phone(p.get("From", ""))}'] = (p.get("CallSid", ""), time.time())


def _stop_ringing(call_sid: str) -> None:
    for key, (sid, _) in list(_ringing_owner.items()):
        if sid == call_sid:
            del _ringing_owner[key]


def owner_number(shop, call_sid: str) -> str:
    """The owner's phone inside a <Dial>, screened with "press 1 to take it"."""
    query = urlencode({"call": call_sid, "shop": shop["id"]})
    url = f"{get_settings().public_base_url}/voice/screen?{query}"
    return f'<Number url="{escape(url)}" method="POST">{escape(shop["owner_phone"])}</Number>'


def dial_owner(shop, caller_id: str, call_sid: str) -> str:
    action = f"{get_settings().public_base_url}/voice/transfer-done"
    return (f'<Dial callerId="{escape(caller_id)}" timeout="25" action="{action}">'
            f'{owner_number(shop, call_sid)}</Dial>')


NOBODY_PICKED_UP = ("Sorry, nobody picked up. We have your details and will call you back as soon as "
                    "possible. Goodbye.")


async def _form(request: Request) -> dict[str, str]:
    form = await request.form()
    params = {k: str(v) for k, v in form.items()}
    s = get_settings()
    if s.twilio_enabled and s.twilio_validate_signatures:
        from twilio.request_validator import RequestValidator

        # Twilio signs the full URL it called, query string included.
        url = f"{s.public_base_url}{request.url.path}" + (f"?{request.url.query}" if request.url.query else "")
        signature = request.headers.get("X-Twilio-Signature", "")
        if not RequestValidator(s.twilio_auth_token).validate(url, params, signature):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")
    return params


@router.post("/voice/incoming")
async def voice_incoming(request: Request):
    p = await _form(request)
    return await run_in_threadpool(_voice_incoming, p)


def _voice_incoming(p: dict[str, str]):
    with db.session() as conn:
        shop = db.shop_by_number(conn, p.get("To", ""))
        if shop is None:
            return twiml(say("Sorry, this number is not in service.") + "<Hangup/>")
        if _is_bounce(shop, p):
            log.info("Call to the owner of shop %s forwarded back to us; answering busy", shop["id"])
            return twiml('<Reject reason="busy"/>')
        if shop["status"] not in ("trial", "active"):
            # Paused or cancelled: pass the call straight through to the owner.
            return twiml(f'<Dial>{escape(shop["owner_phone"])}</Dial>')
        # "after_hours" mode (always on Starter): during business hours ring the owner first,
        # and the assistant picks up only if they don't answer.
        ring_owner_first = shop["answer_mode"] == "after_hours" or not get_plan(shop["plan"]).answers_24_7
        if ring_owner_first and scheduling.is_open_now(shop):
            _start_ringing(shop, p)
            return twiml(f'<Dial timeout="20" action="{get_settings().public_base_url}/voice/no-answer">'
                         f'{owner_number(shop, p.get("CallSid", ""))}</Dial>')
        if calls.over_cap(conn, shop):
            log.warning("Shop %s is over its monthly call cap; passing call through", shop["id"])
            return twiml(f'<Dial>{escape(shop["owner_phone"])}</Dial>')
        calls.start_call(conn, shop, p.get("CallSid", ""), p.get("From", ""))
        return twiml(gather(greeting(shop)))


@router.post("/voice/no-answer")
async def voice_no_answer(request: Request):
    """Owner didn't pick up during business hours: the assistant takes over."""
    p = await _form(request)
    return await run_in_threadpool(_voice_no_answer, p)


def _voice_no_answer(p: dict[str, str]):
    _stop_ringing(p.get("CallSid", ""))
    # "completed" only means *something* picked up; it was the owner only if they pressed 1.
    if _was_accepted(p.get("CallSid", "")) and p.get("DialCallStatus") == "completed":
        return twiml("<Hangup/>")
    with db.session() as conn:
        shop = db.shop_by_number(conn, p.get("To", ""))
        if shop is None:
            return twiml("<Hangup/>")
        calls.start_call(conn, shop, p.get("CallSid", ""), p.get("From", ""))
        return twiml(gather(greeting(shop)))


@router.post("/voice/respond")
async def voice_respond(request: Request):
    p = await _form(request)
    return await run_in_threadpool(_voice_respond, p)


def _voice_respond(p: dict[str, str]):
    with db.session() as conn:
        call = conn.execute("SELECT * FROM calls WHERE call_sid = ?", (p.get("CallSid", ""),)).fetchone()
        if call is None:
            return twiml("<Hangup/>")
        shop = db.get(conn, "shops", call["shop_id"])
        turn = Receptionist(conn, shop, call).respond(p.get("SpeechResult", ""))
        if turn.action == "transfer":
            calls.finalize_call(conn, call["id"], status="transferred")
            return twiml(say(turn.say) + dial_owner(shop, shop["twilio_number"], p.get("CallSid", "")))
        if turn.action == "hangup":
            calls.finalize_call(conn, call["id"])
            return twiml(say(turn.say) + "<Hangup/>")
        return twiml(gather(turn.say))


@router.post("/voice/screen")
async def voice_screen(request: Request, call: str = "", shop: int = 0):
    """Played to the owner when their phone is answered, before the caller is connected."""
    await _form(request)
    with db.session() as conn:
        row = db.get(conn, "shops", shop)
    name = row["name"] if row else "your shop"
    action = f"{get_settings().public_base_url}/voice/screen-result?{urlencode({'call': call})}"
    prompt = say(f"Call for {name} from your assistant. Press 1 to take it.")
    # One short prompt: the caller hears ringing until the owner presses 1 or this hangs up.
    return twiml(f'<Gather numDigits="1" timeout="5" action="{escape(action)}" method="POST">'
                 f'{prompt}</Gather><Hangup/>')


@router.post("/voice/screen-result")
async def voice_screen_result(request: Request, call: str = ""):
    p = await _form(request)
    if p.get("Digits") == "1" and call:
        _mark_accepted(call)
        return twiml("")  # an empty response connects the owner to the caller
    return twiml("<Hangup/>")


@router.post("/voice/transfer-done")
async def voice_transfer_done(request: Request):
    """After a transfer: if the owner never really took it, tell the caller we'll call back."""
    p = await _form(request)
    if _was_accepted(p.get("CallSid", "")) and p.get("DialCallStatus") == "completed":
        return twiml("<Hangup/>")
    return twiml(say(NOBODY_PICKED_UP) + "<Hangup/>")


@router.post("/voice/status")
async def voice_status(request: Request):
    """Twilio call status callback: catches callers who hang up mid-conversation."""
    p = await _form(request)
    return await run_in_threadpool(_voice_status, p)


def _voice_status(p: dict[str, str]):
    if p.get("CallStatus") in ("completed", "busy", "failed", "no-answer", "canceled"):
        _stop_ringing(p.get("CallSid", ""))
        with db.session() as conn:
            call = conn.execute("SELECT id FROM calls WHERE call_sid = ?", (p.get("CallSid", ""),)).fetchone()
            if call:
                calls.finalize_call(conn, call["id"])
    return Response(status_code=204)


OWNER_HELP = ("Commands: DONE <appt#> marks a job complete and sends a review request. "
              "QUOTE <phone> <amount> <description> starts quote follow-ups. PAUSE / RESUME the assistant.")


@router.post("/sms/incoming")
async def sms_incoming(request: Request):
    p = await _form(request)
    return await run_in_threadpool(_sms_incoming, p)


def _sms_incoming(p: dict[str, str]):
    sender, body = p.get("From", ""), p.get("Body", "").strip()
    with db.session() as conn:
        shop = db.shop_by_number(conn, p.get("To", ""))
        if shop is None:
            return twiml("")
        db.insert(conn, "sms_log", {"shop_id": shop["id"], "direction": "inbound", "to_phone": p.get("To", ""),
                                    "from_phone": sender, "body": body, "created_at": db.iso(db.utcnow())})
        if normalize_phone(sender) == normalize_phone(shop["owner_phone"]):
            return twiml(f"<Message>{escape(handle_owner_command(conn, shop, body))}</Message>")

        word = body.upper()
        if word in ("STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT"):
            conn.execute("INSERT OR IGNORE INTO opt_outs (shop_id, phone, created_at) VALUES (?, ?, ?)",
                         (shop["id"], sender, db.iso(db.utcnow())))
            followups.cancel_for_phone(conn, shop["id"], sender,
                                       ("unbooked_caller", "quote", "review_request", "maintenance"))
            return twiml("")  # Twilio sends the carrier-required opt-out confirmation itself
        if word in ("START", "UNSTOP"):
            conn.execute("DELETE FROM opt_outs WHERE shop_id = ? AND phone = ?", (shop["id"], sender))
            return twiml("")
        # Any customer reply: stop automated nudges and hand the conversation to the owner.
        followups.cancel_for_phone(conn, shop["id"], sender, ("unbooked_caller", "quote"))
        sms.send(conn, shop, shop["owner_phone"], f"Text from customer {sender}: {body}", to_owner=True)
        reply = f"Thanks! Someone from {shop['name']} will get back to you shortly."
        return twiml(f"<Message>{escape(reply)}</Message>")


def handle_owner_command(conn, shop, body: str) -> str:
    parts = body.split()
    cmd = parts[0].upper() if parts else ""
    if cmd == "DONE" and len(parts) >= 2 and parts[1].lstrip("#").isdigit():
        appt = db.get(conn, "appointments", int(parts[1].lstrip("#")))
        if appt is None or appt["shop_id"] != shop["id"]:
            return "No appointment with that number."
        calls.complete_appointment(conn, appt["id"])
        return f"Marked #{appt['id']} done. Review request queued."
    if cmd == "QUOTE" and len(parts) >= 4:
        phone = normalize_phone(parts[1])
        try:
            amount = float(parts[2].replace("$", "").replace(",", ""))
        except ValueError:
            return "Format: QUOTE 5125550123 1850 water heater replacement"
        if not phone:
            return "That phone number doesn't look right."
        qid = db.insert(conn, "quotes", {"shop_id": shop["id"], "customer_phone": phone, "amount": amount,
                                         "description": " ".join(parts[3:]), "created_at": db.iso(db.utcnow())})
        n = len(followups.schedule_quote_followups(conn, shop, db.get(conn, "quotes", qid), now=db.utcnow()))
        return f"Quote #{qid} saved. {n} follow-up texts scheduled." if n else \
            f"Quote #{qid} saved. Quote follow-ups are on the Plus plan."
    if cmd in ("PAUSE", "RESUME"):
        resumed = "active" if shop["stripe_subscription_id"] else "trial"
        db.update(conn, "shops", shop["id"], {"status": "paused" if cmd == "PAUSE" else resumed})
        return "Assistant paused; calls ring straight to you." if cmd == "PAUSE" else "Assistant back on."
    return OWNER_HELP
