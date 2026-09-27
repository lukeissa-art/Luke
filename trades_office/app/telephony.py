"""Twilio webhooks: voice (answer, converse, transfer) and inbound SMS.

Setup per shop: buy a Twilio number, point its Voice webhook at /voice/incoming and its
Messaging webhook at /sms/incoming. The shop forwards calls to that number (all calls for
24/7 plans, or no-answer/after-hours forwarding for Starter).
"""

import logging
from xml.sax.saxutils import escape

from fastapi import APIRouter, HTTPException, Request, Response

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


def dial_owner(shop, caller_id: str) -> str:
    return (f'<Dial callerId="{escape(caller_id)}" timeout="25">{escape(shop["owner_phone"])}</Dial>'
            + say("Sorry, nobody picked up. We have your details and will call you back as soon as "
                  "possible. Goodbye.")
            + "<Hangup/>")


async def _form(request: Request) -> dict[str, str]:
    form = await request.form()
    params = {k: str(v) for k, v in form.items()}
    s = get_settings()
    if s.twilio_enabled and s.twilio_validate_signatures:
        from twilio.request_validator import RequestValidator

        url = f"{s.public_base_url}{request.url.path}"
        signature = request.headers.get("X-Twilio-Signature", "")
        if not RequestValidator(s.twilio_auth_token).validate(url, params, signature):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")
    return params


@router.post("/voice/incoming")
async def voice_incoming(request: Request):
    p = await _form(request)
    with db.session() as conn:
        shop = db.shop_by_number(conn, p.get("To", ""))
        if shop is None:
            return twiml(say("Sorry, this number is not in service.") + "<Hangup/>")
        if shop["status"] not in ("trial", "active"):
            # Paused or cancelled: pass the call straight through to the owner.
            return twiml(f'<Dial>{escape(shop["owner_phone"])}</Dial>')
        # "after_hours" mode (always on Starter): during business hours ring the owner first,
        # and the assistant picks up only if they don't answer.
        ring_owner_first = shop["answer_mode"] == "after_hours" or not get_plan(shop["plan"]).answers_24_7
        if ring_owner_first and scheduling.is_open_now(shop):
            return twiml(f'<Dial timeout="20" action="{get_settings().public_base_url}/voice/no-answer">'
                         f'{escape(shop["owner_phone"])}</Dial>')
        if calls.over_cap(conn, shop):
            log.warning("Shop %s is over its monthly call cap; passing call through", shop["id"])
            return twiml(f'<Dial>{escape(shop["owner_phone"])}</Dial>')
        calls.start_call(conn, shop, p.get("CallSid", ""), p.get("From", ""))
        return twiml(gather(greeting(shop)))


@router.post("/voice/no-answer")
async def voice_no_answer(request: Request):
    """Owner didn't pick up during business hours: the assistant takes over."""
    p = await _form(request)
    if p.get("DialCallStatus") == "completed":
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
    with db.session() as conn:
        call = conn.execute("SELECT * FROM calls WHERE call_sid = ?", (p.get("CallSid", ""),)).fetchone()
        if call is None:
            return twiml("<Hangup/>")
        shop = db.get(conn, "shops", call["shop_id"])
        turn = Receptionist(conn, shop, call).respond(p.get("SpeechResult", ""))
        if turn.action == "transfer":
            calls.finalize_call(conn, call["id"], status="transferred")
            return twiml(say(turn.say) + dial_owner(shop, shop["twilio_number"]))
        if turn.action == "hangup":
            calls.finalize_call(conn, call["id"])
            return twiml(say(turn.say) + "<Hangup/>")
        return twiml(gather(turn.say))


@router.post("/voice/status")
async def voice_status(request: Request):
    """Twilio call status callback: catches callers who hang up mid-conversation."""
    p = await _form(request)
    if p.get("CallStatus") in ("completed", "busy", "failed", "no-answer", "canceled"):
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
