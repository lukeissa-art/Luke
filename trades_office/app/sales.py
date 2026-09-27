"""Website assistant: answers visitors' questions about the company (plans, trial, setup)."""

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .plans import PLANS, SETUP_FEE, TRIAL_DAYS
from .receptionist import make_backend

log = logging.getLogger(__name__)
router = APIRouter()

MAX_HISTORY = 12
MAX_CHARS = 500


def _plan_lines() -> str:
    lines = []
    for p in PLANS.values():
        features = [f"up to {p.monthly_call_cap} calls a month",
                    "answers 24/7" if p.answers_24_7 else "answers after-hours and missed calls only",
                    "a text summary of every call", "emergency transfer to the owner's cell"]
        features += [label for flag, label in (
            (p.booking, "books appointments into the calendar"),
            (p.followups, "follow-up texts to callers who didn't book"),
            (p.review_requests, "Google review requests after jobs"),
            (p.quote_followups, "automatic quote follow-up texts"),
            (p.maintenance_reminders, "maintenance reminder texts (e.g. HVAC tune-ups)"),
            (p.multi_location, "multiple locations"),
        ) if flag]
        lines.append(f"- {p.name}: ${p.monthly_price}/month. " + "; ".join(features) + ".")
    return "\n".join(lines)


def system_prompt() -> str:
    s = get_settings()
    return f"""You are the assistant on the website of {s.company_name}. You answer questions from owners of \
plumbing, HVAC and electrical shops about {s.company_name}. Be friendly, plain-spoken and brief: two to four short \
sentences, no markdown headings. Use ONLY the facts below. If you don't know something, say so and suggest the free \
trial form on this page or emailing {s.contact_email}. Never invent prices, discounts, features, integrations or \
customer numbers. When it fits, invite them to start the free trial (the form is further down this page) or to try \
the demo call on this page.

What {s.company_name} is:
- An AI receptionist for small trades shops. It answers the shop's calls in the shop's name, collects the \
caller's name, number, address and problem, books jobs into open arrival windows, and texts the owner a summary \
of every call.
- Emergencies follow fixed rules: gas smell, carbon monoxide, sparks or smoke get safety instructions and an \
immediate transfer to the owner's cell. Flooding, burst pipes, sewage and no heat get an urgent text to the owner \
and an offer to transfer. If the AI ever fails, the call goes to the owner.
- It only quotes prices the owner approves; otherwise it says the technician will price the job on site.
- Every call starts with a notice that the call may be recorded and that it's the shop's virtual assistant.
- Owners keep their existing phone number and forward calls to a line we provide (all calls, or just after-hours \
and missed calls). They can text PAUSE to switch it off and RESUME to switch it back on.
- Serving shops in {s.service_region}.

Plans (month to month, cancel anytime, no contract):
{_plan_lines()}
- One-time setup fee: ${SETUP_FEE}, waived for our first 10 shops in exchange for a testimonial.
- Over the monthly call limit, calls simply ring through to the owner; nothing is cut off.

Free trial:
- {TRIAL_DAYS} days, free, no card needed. During the trial it answers after-hours calls only, so nothing changes \
during the workday. At the end we go through every call it caught and every job it booked together, and the owner \
decides whether to keep it.
- Setup is done for them: about two hours with us, in person or over video, to learn their services, service area, \
prices and emergency rules, then test calls together.

Integrations: books into Google Calendar or its own built-in schedule today. Jobber and Housecall Pro are planned \
but not available yet.

Contact: {s.contact_email}
"""


@router.post("/ask")
async def ask(request: Request):
    from .site import _client_ip, _rate_limited

    if _rate_limited("ask:" + _client_ip(request)):
        return JSONResponse({"error": "You've asked a lot of questions. Try again in a bit, or email us."},
                            status_code=429)
    body = await request.json()
    return await run_in_threadpool(_answer, body)


def _answer(body: dict) -> Any:
    history = body.get("history") or []
    question = str(body.get("question", "")).strip()[:MAX_CHARS]
    if not question:
        return JSONResponse({"error": "Type a question first."}, status_code=400)
    backend = make_backend()
    messages = []
    for turn in history[-MAX_HISTORY:]:
        text = str(turn.get("text", ""))[:MAX_CHARS]
        if text:
            messages.append(backend.user_message(text) if turn.get("role") == "user"
                            else backend.assistant_message(text))
    # Conversations must start with the visitor.
    while messages and messages[0].get("role") not in ("user",):
        messages.pop(0)
    messages.append(backend.user_message(question))
    try:
        step = backend.step(system_prompt(), [], messages)
    except Exception as exc:
        log.exception("Website assistant failed (%s: %s)", type(exc).__name__, str(exc)[:300])
        return JSONResponse({"answer": "Sorry, I can't answer right now. Please email "
                                       f"{get_settings().contact_email} or use the trial form below."})
    if step.refused or not step.spoken:
        return JSONResponse({"answer": "I can't help with that one, but the team can: "
                                       f"email {get_settings().contact_email}."})
    return JSONResponse({"answer": " ".join(step.spoken).strip()})
