"""The AI receptionist: one phone conversation, driven by Claude with a small set of tools.

Each caller utterance arrives as a separate webhook, so conversation state (the Claude
message history) is stored on the call row between turns.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import anthropic

from . import db, emergency, scheduling, sms
from .config import get_settings
from .plans import get_plan

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5
TRADE_LABELS = {"plumbing": "plumbing", "hvac": "heating and air", "electrical": "electrical"}


@dataclass
class Turn:
    say: str
    action: str = "continue"  # "continue" | "transfer" | "hangup"


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


_STR = {"type": "string"}

TOOL_CHECK_AVAILABILITY = _tool(
    "check_availability",
    "Look up open arrival windows on the shop's schedule. Call this before offering any times.",
    {
        "preferred_date": {
            "type": "string",
            "description": "YYYY-MM-DD if the caller asked for a specific day, otherwise an empty string "
            "for the next available windows.",
        }
    },
)
TOOL_BOOK = _tool(
    "book_appointment",
    "Book the caller into an arrival window returned by check_availability. Only call after the caller "
    "has picked a window and you have their name, callback number, service address and problem.",
    {
        "slot_start": {**_STR, "description": "The exact 'start' value from check_availability."},
        "customer_name": _STR,
        "callback_phone": {**_STR, "description": "Digits only, e.g. 5125550123."},
        "address": {**_STR, "description": "Full service address as confirmed with the caller."},
        "problem_summary": {**_STR, "description": "One sentence describing the job."},
    },
)
TOOL_RECORD = _tool(
    "record_caller_details",
    "Save the caller's details so the owner can call back. Call this as soon as you have them, even if "
    "you also book an appointment.",
    {
        "caller_name": _STR,
        "callback_phone": {**_STR, "description": "Digits only."},
        "address": _STR,
        "problem_summary": _STR,
        "is_urgent": {"type": "boolean"},
    },
)
TOOL_TRANSFER = _tool(
    "transfer_to_owner",
    "Connect the caller to the owner's cell phone right now. Use for emergencies, when the caller "
    "insists on a person, or for anything you cannot handle (billing disputes, complaints, vendors).",
    {"reason": _STR},
)
TOOL_END = _tool(
    "end_call",
    "Hang up after you have said goodbye. Always include a short summary for the owner.",
    {
        "outcome": {
            "type": "string",
            "enum": ["booked", "message_taken", "not_a_customer", "spam", "caller_done"],
        },
        "summary": {**_STR, "description": "Two sentences max, written for the shop owner."},
    },
)


def build_system_prompt(shop: sqlite3.Row) -> str:
    s = get_settings()
    plan = get_plan(shop["plan"])
    trade = TRADE_LABELS.get(shop["trade"], shop["trade"])
    booking_rule = (
        "You can book appointments. Offer at most two arrival windows at a time, read them naturally, "
        "and confirm the chosen window, name, number and address back before booking."
        if plan.booking
        else "You cannot book appointments on this plan. Take a detailed message and tell the caller "
        "someone from the shop will call them back shortly."
    )
    return f"""You are the phone receptionist for {shop['name']}, a {trade} company. \
You answer on the shop's behalf when the team is on job sites or after hours. \
You are a virtual assistant provided by {s.company_name}; if anyone asks whether you are a person or AI, say you \
are the shop's virtual assistant.

Your goal on every call: capture the job. Get the caller's name, the best callback number, the service \
address, and a clear description of the problem, then book them or take a message.

How to talk:
- This is a phone call. Your words are read aloud by a text-to-speech voice. Speak in one or two short, \
warm sentences at a time. No lists, no markdown, no emoji, no abbreviations like "approx."
- Ask one question at a time. Speech recognition makes mistakes, so repeat back addresses and phone \
numbers to confirm them. Read phone numbers digit by digit.
- Use the caller ID number if the caller says it's the best number to reach them.
- Never diagnose the problem or give repair instructions. The only exception is safety: water shut-off \
valve for an active leak, or leaving the building for gas or smoke.
- Never quote a price unless it appears in the pricing notes below. Otherwise say the technician will \
give an exact price on site.
- Never promise an exact arrival time; appointments are arrival windows.
- If someone is selling something, asking for a job, or is clearly spam, politely end the call.
- {booking_rule}
- Emergencies (gas smell, carbon monoxide, sparks or smoke, flooding, sewage backup, no heat in freezing \
weather, anyone vulnerable at risk): safety instructions first, then use transfer_to_owner right away.
- Before you end the call, tell the caller what happens next and say goodbye, then call end_call.

Shop details:
- Service area: {shop['service_area'] or 'ask the owner; if unsure, take the details and say the shop will confirm'}
- Services: {shop['services'] or f'general {trade} repair and installation'}
- Pricing notes: {shop['pricing_notes'] or 'none; do not quote prices'}
- Owner: {shop['owner_name'] or 'the owner'}
{('- Extra instructions from the owner: ' + shop['extra_instructions']) if shop['extra_instructions'] else ''}
"""


def greeting(shop: sqlite3.Row) -> str:
    return (
        f"Thanks for calling {shop['name']}. This call may be recorded. "
        "I'm the shop's virtual assistant. How can I help you today?"
    )


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _block_dict(block: Any) -> dict[str, Any]:
    return block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else dict(block)


class Receptionist:
    def __init__(
        self,
        conn: sqlite3.Connection,
        shop: sqlite3.Row,
        call: sqlite3.Row,
        *,
        client: Any = None,
        now: datetime | None = None,
    ):
        self.conn = conn
        self.shop = shop
        self.call_id = call["id"]
        self.caller_phone = call["caller_phone"]
        self.is_demo = call["call_sid"].startswith("demo-")
        self.is_public_demo = call["call_sid"].startswith("demo-web-")
        self.messages: list[dict[str, Any]] = json.loads(call["messages"])
        self.transcript: list[dict[str, str]] = json.loads(call["transcript"])
        self.client = client or _client()
        self.now = now
        self.plan = get_plan(shop["plan"])
        self.action = "continue"
        self.end_state: dict[str, str] = {}

    # --- public -------------------------------------------------------------------------

    def respond(self, caller_text: str) -> Turn:
        caller_text = caller_text.strip()
        self.transcript.append({"role": "caller", "text": caller_text})
        content = caller_text or "(the caller said nothing or was inaudible)"
        if not self.messages:
            tz = ZoneInfo(self.shop["timezone"])
            local = (self.now or db.utcnow()).astimezone(tz)
            content = (
                f"[Call started {local:%A %B %d %Y, %I:%M %p} shop local time. "
                f"Caller ID: {self.caller_phone or 'unknown'}. "
                f"You already greeted them: \"{greeting(self.shop)}\"]\n\n{content}"
            )

        match = emergency.check(caller_text, self.shop["emergency_keywords"])
        if match and match.level == "life_safety":
            return self._life_safety(match, content)
        if match:
            self._flag_urgent(match.keyword)
            content += (
                f"\n\n[System: urgent keyword detected ({match.keyword}). The owner has been texted. "
                "Offer to connect the caller to the owner right now.]"
            )

        self.messages.append({"role": "user", "content": content})
        try:
            say = self._run_model()
        except Exception:  # never leave a live caller hanging: any AI failure goes to a human
            log.exception("Assistant failed on call %s; transferring to owner", self.call_id)
            say = "I'm sorry, I'm having trouble on my end. Let me connect you to the team."
            self.action = "transfer"
            self.end_state = {"outcome": "transferred", "summary": "Assistant error; call sent to you."}
            self.messages.append({"role": "assistant", "content": say})
        self.transcript.append({"role": "ai", "text": say})
        self._save()
        return Turn(say=say, action=self.action)

    # --- model loop ---------------------------------------------------------------------

    def _tools(self) -> list[dict[str, Any]]:
        tools = [TOOL_RECORD, TOOL_TRANSFER, TOOL_END]
        if self.plan.booking:
            tools = [TOOL_CHECK_AVAILABILITY, TOOL_BOOK, *tools]
        return tools

    def _create(self):
        s = get_settings()
        kwargs: dict[str, Any] = {}
        if s.anthropic_fallbacks:
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        return self.client.beta.messages.create(
            model=s.anthropic_model,
            max_tokens=2048,
            system=build_system_prompt(self.shop),
            tools=self._tools(),
            messages=self.messages,
            output_config={"effort": s.anthropic_effort},
            cache_control={"type": "ephemeral"},
            **kwargs,
        )

    def _run_model(self) -> str:
        spoken: list[str] = []
        for _ in range(MAX_TOOL_ROUNDS):
            response = self._create()
            if response.stop_reason == "refusal":
                # Declined even after server-side fallback: end politely and have the owner call back.
                self.end_state = {"outcome": "message_taken", "summary": "Caller needs a callback."}
                self.action = "hangup"
                return "Thanks for calling. Someone from the shop will call you back shortly. Goodbye."

            blocks = [_block_dict(b) for b in response.content if b.type != "fallback"]
            self.messages.append({"role": "assistant", "content": blocks})
            spoken += [b.text for b in response.content if b.type == "text" and b.text.strip()]

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                break
            results = [self._run_tool(t.name, t.input, t.id) for t in tool_uses]
            self.messages.append({"role": "user", "content": results})
            if self.action != "continue" and spoken:
                break

        say = " ".join(spoken).strip()
        if not say:
            say = {
                "transfer": "One moment, I'm connecting you now.",
                "hangup": "Thanks for calling. Goodbye.",
            }.get(self.action, "Sorry, could you say that one more time?")
        return say

    def _run_tool(self, name: str, args: dict[str, Any], tool_use_id: str) -> dict[str, Any]:
        try:
            result = getattr(self, f"_tool_{name}")(**args)
            is_error = False
        except scheduling.SlotUnavailable:
            result = "That window was just taken or is not valid. Call check_availability again."
            is_error = True
        except Exception as exc:  # report tool failures back to the model instead of crashing the call
            log.exception("Tool %s failed", name)
            result = f"Tool error: {exc}"
            is_error = True
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result if isinstance(result, str) else json.dumps(result),
            "is_error": is_error,
        }

    # --- tools --------------------------------------------------------------------------

    def _tool_check_availability(self, preferred_date: str) -> Any:
        on_date = None
        if preferred_date:
            on_date = datetime.fromisoformat(preferred_date).date()
        slots = scheduling.open_slots(self.conn, self.shop, now=self.now, on_date=on_date)
        if not slots and on_date:
            return {"slots": [], "note": "Nothing open that day. Try an empty preferred_date."}
        return {"slots": slots}

    def _tool_book_appointment(
        self, slot_start: str, customer_name: str, callback_phone: str, address: str, problem_summary: str
    ) -> str:
        phone = normalize_phone(callback_phone) or self.caller_phone
        appt_id = scheduling.book(
            self.conn,
            self.shop,
            start_iso=slot_start,
            customer_name=customer_name,
            customer_phone=phone,
            address=address,
            problem=problem_summary,
            call_id=self.call_id,
            now=self.now,
            sync_external=not self.is_demo,
        )
        self._record_details(
            {"caller_name": customer_name, "callback_phone": phone, "address": address,
             "problem_summary": problem_summary}
        )
        when = scheduling.label(self.shop, slot_start)
        if not self.is_demo:
            sms.send(
                self.conn,
                self.shop,
                phone,
                f"{self.shop['name']}: you're booked for {when} at {address}. "
                "We'll text when the tech is on the way. Reply STOP to opt out.",
            )
        db.update(self.conn, "calls", self.call_id, {"outcome": "booked"})
        return f"Booked (appointment #{appt_id}) for {when}. A confirmation text was sent."

    def _tool_record_caller_details(
        self, caller_name: str, callback_phone: str, address: str, problem_summary: str, is_urgent: bool
    ) -> str:
        self._record_details(
            {"caller_name": caller_name, "callback_phone": callback_phone, "address": address,
             "problem_summary": problem_summary}
        )
        if is_urgent:
            self._flag_urgent(problem_summary)
        return "Saved."

    def _tool_transfer_to_owner(self, reason: str) -> str:
        self.action = "transfer"
        self.end_state = {"outcome": "transferred", "summary": reason}
        return "Transfer queued. Tell the caller you're connecting them now; do not ask more questions."

    def _tool_end_call(self, outcome: str, summary: str) -> str:
        self.action = "hangup"
        self.end_state = {"outcome": outcome, "summary": summary}
        return "Call will end after your goodbye."

    # --- helpers ------------------------------------------------------------------------

    def _record_details(self, details: dict[str, str]) -> None:
        values = {}
        if details.get("caller_name"):
            values["caller_name"] = details["caller_name"]
        if details.get("address"):
            values["address"] = details["address"]
        if details.get("problem_summary"):
            values["problem"] = details["problem_summary"]
        phone = normalize_phone(details.get("callback_phone", ""))
        if phone:
            values["callback_phone"] = phone
        if values:
            db.update(self.conn, "calls", self.call_id, values)

    def _flag_urgent(self, what: str) -> None:
        call = db.get(self.conn, "calls", self.call_id)
        if call["is_emergency"]:
            return
        db.update(self.conn, "calls", self.call_id, {"is_emergency": 1})
        if self.is_public_demo:
            return
        sms.send(
            self.conn,
            self.shop,
            self.shop["owner_phone"],
            f"URGENT call now from {pretty_phone(self.caller_phone)}: {what}. "
            "The assistant is offering to transfer them to you.",
            to_owner=True,
        )

    def _life_safety(self, match: emergency.EmergencyMatch, content: str) -> Turn:
        self._flag_urgent(match.keyword)
        say = f"{match.caller_instructions} I'm connecting you to the owner right now."
        self.messages.append({"role": "user", "content": content})
        self.messages.append({"role": "assistant", "content": say})
        self.transcript.append({"role": "ai", "text": say})
        self.action = "transfer"
        self.end_state = {"outcome": "transferred", "summary": f"EMERGENCY ({match.keyword}): {content[-200:]}"}
        self._save()
        return Turn(say=say, action="transfer")

    def _save(self) -> None:
        values: dict[str, Any] = {
            "messages": json.dumps(self.messages),
            "transcript": json.dumps(self.transcript),
        }
        if self.end_state.get("summary"):
            values["summary"] = self.end_state["summary"]
        if self.end_state.get("outcome"):
            call = db.get(self.conn, "calls", self.call_id)
            if call["outcome"] != "booked":
                values["outcome"] = self.end_state["outcome"]
        db.update(self.conn, "calls", self.call_id, values)


def normalize_phone(raw: str) -> str:
    digits = "".join(c for c in raw or "" if c.isdigit())
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return ""


def pretty_phone(e164: str) -> str:
    d = "".join(c for c in e164 or "" if c.isdigit())[-10:]
    return f"({d[:3]}) {d[3:6]}-{d[6:]}" if len(d) == 10 else (e164 or "unknown number")
