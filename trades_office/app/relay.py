"""Natural phone conversations over Twilio ConversationRelay (turn on with VOICE_ENGINE=relay).

Instead of "AI talks, then listens" (<Gather>), Twilio keeps a WebSocket open for the whole call:
it transcribes the caller continuously, handles echo (speakerphone), lets the caller interrupt
the AI mid-sentence, and speaks our reply as the model writes it. The conversation itself, with
its tools, booking and emergency rules, is the same Receptionist the <Gather> line and the
website chat use.

Flow: /voice/incoming answers with <Connect><ConversationRelay url="wss://.../voice/relay">.
Twilio sends "prompt" messages (what the caller said) and "interrupt" messages; we send "text"
tokens back. To transfer or hang up we send "end" with handoffData, and Twilio then requests
/voice/relay-done, which returns the TwiML for what happens next.
"""

import asyncio
import hashlib
import hmac
import json
import logging
from urllib.parse import urlencode
from xml.sax.saxutils import quoteattr

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from . import calls, db
from .config import get_settings
from .receptionist import Receptionist, greeting

log = logging.getLogger(__name__)
router = APIRouter()

ENDED = ("completed", "canceled", "failed", "busy", "no-answer")
WORDS_PER_SECOND = 2.6  # roughly how fast the voice speaks, to let a goodbye finish before ending


def enabled() -> bool:
    return get_settings().voice_engine == "relay"


def _token(call_sid: str) -> str:
    """Per-call secret in the WebSocket URL. Only Twilio sees it (in TwiML from a signed request),
    so nobody else can open a session for a call."""
    s = get_settings()
    key = (s.twilio_auth_token or s.admin_password).encode()
    return hmac.new(key, f"relay:{call_sid}".encode(), hashlib.sha256).hexdigest()[:32]


def relay_twiml(shop, call_sid: str) -> str:
    s = get_settings()
    scheme, host = s.public_base_url.split("://", 1)
    ws_url = (f"{'wss' if scheme == 'https' else 'ws'}://{host}/voice/relay?"
              f"{urlencode({'call': call_sid, 'token': _token(call_sid)})}")
    attrs = {"url": ws_url, "welcomeGreeting": greeting(shop), "language": "en-US"}
    for name, value in (("ttsProvider", s.relay_tts_provider), ("voice", s.relay_voice),
                        ("transcriptionProvider", s.relay_transcription_provider)):
        if value:
            attrs[name] = value
    attr_xml = " ".join(f"{name}={quoteattr(value)}" for name, value in attrs.items())
    return f'<Connect action="{s.public_base_url}/voice/relay-done"><ConversationRelay {attr_xml}/></Connect>'


def _speaking_time(text: str) -> float:
    words = len(text.split())
    return min(15.0, words / WORDS_PER_SECOND + 1.0) if words else 0.0


def _norm(text: str) -> str:
    return " ".join(text.split())


def _run_turn(call_sid: str, text: str, on_text):
    """One caller turn through the Receptionist. Returns the Turn, or None if the call is unknown."""
    with db.session() as conn:
        call = conn.execute("SELECT * FROM calls WHERE call_sid = ?", (call_sid,)).fetchone()
        if call is None:
            return None
        shop = db.get(conn, "shops", call["shop_id"])
        turn = Receptionist(conn, shop, call).respond(text, on_text)
        if turn.action == "transfer":
            calls.finalize_call(conn, call["id"], status="transferred")
        elif turn.action == "hangup":
            calls.finalize_call(conn, call["id"])
        return turn


class _Session:
    """One live call. Messages to Twilio go through a single queue so they stay in order."""

    def __init__(self, ws: WebSocket, call_sid: str):
        self.ws, self.call_sid = ws, call_sid
        self.loop = asyncio.get_running_loop()
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()  # one turn at a time, in order
        self.turn = 0  # bumped by each prompt and interrupt; text from an older turn is dropped
        self.partial = ""
        self.closed = False

    def send(self, msg: dict) -> None:
        self.outbox.put_nowait(msg)

    def send_from_thread(self, msg: dict) -> None:
        self.loop.call_soon_threadsafe(self.outbox.put_nowait, msg)

    async def sender(self) -> None:
        while (msg := await self.outbox.get()) is not None:
            try:
                await self.ws.send_text(json.dumps(msg))
            except Exception:
                self.closed = True
                return

    async def handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "setup":
            if msg.get("callSid") and msg["callSid"] != self.call_sid:
                log.warning("Relay setup for %s on a session for %s; closing", msg["callSid"], self.call_sid)
                self.closed = True
                await self.ws.close(code=1008)
        elif kind == "prompt":
            self.partial += msg.get("voicePrompt", "")
            if msg.get("last", True):
                text, self.partial = self.partial, ""
                self.turn += 1
                asyncio.create_task(self.answer(self.turn, text))
        elif kind == "interrupt":
            self.turn += 1  # the caller cut in: Twilio stopped speaking, so stop sending the rest
        elif kind == "error":
            log.warning("ConversationRelay error on %s: %s", self.call_sid, msg.get("description"))

    async def answer(self, turn_no: int, text: str) -> None:
        async with self.lock:
            if self.closed:
                return
            streamed: list[str] = []

            def on_text(piece: str) -> None:
                if turn_no == self.turn and not self.closed:
                    streamed.append(piece)
                    self.send_from_thread({"type": "text", "token": piece, "last": False})

            try:
                turn = await asyncio.to_thread(_run_turn, self.call_sid, text, on_text)
            except Exception:
                log.exception("Relay turn failed on %s; handing the call to the owner", self.call_sid)
                self.send({"type": "end", "handoffData": json.dumps({"action": "trouble"})})
                return
            if turn is None:
                self.send({"type": "end", "handoffData": json.dumps({"action": "hangup"})})
                return

            spoken = "".join(streamed)
            if turn_no == self.turn:
                # The final reply can differ from what streamed (a fixed emergency script, or the
                # model failing partway): make sure the caller hears it.
                extra = "" if _norm(turn.say) == _norm(spoken) else (" " if spoken else "") + turn.say
                self.send({"type": "text", "token": extra, "last": True})
                spoken += extra
            if turn.action in ("transfer", "hangup"):
                # Let the last words finish playing before the session ends.
                await asyncio.sleep(_speaking_time(spoken if turn_no == self.turn else ""))
                self.send({"type": "end", "handoffData": json.dumps({"action": turn.action})})


@router.websocket("/voice/relay")
async def voice_relay(ws: WebSocket, call: str = "", token: str = ""):
    if not call or not hmac.compare_digest(token, _token(call)):
        await ws.close(code=1008)
        return
    await ws.accept()
    session = _Session(ws, call)
    sender = asyncio.create_task(session.sender())
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            await session.handle(msg)
            if session.closed:
                break
    except WebSocketDisconnect:
        pass
    finally:
        session.closed = True
        session.outbox.put_nowait(None)
        await sender


@router.post("/voice/relay-done")
async def voice_relay_done(request: Request):
    """The relay session ended: transfer, hang up, or rescue the call if the relay failed."""
    from .telephony import _form

    p = await _form(request)
    return await run_in_threadpool(_relay_done, p)


def _relay_done(p: dict[str, str]):
    from .telephony import TROUBLE, dial_owner, say, twiml

    if p.get("CallStatus") in ENDED:
        return twiml("<Hangup/>")
    try:
        handoff = json.loads(p.get("HandoffData") or "{}")
    except ValueError:
        handoff = {}
    action = handoff.get("action") if isinstance(handoff, dict) else None
    with db.session() as conn:
        shop = db.shop_by_number(conn, p.get("To", ""))
        if shop is None or action == "hangup":
            return twiml("<Hangup/>")
        if action == "transfer":
            return twiml(dial_owner(shop, shop["twilio_number"], p.get("CallSid", "")))
        # Ended without a handoff (relay error) or on a failed turn: never drop the caller.
        log.error("Relay session on %s ended without a handoff (%s); sending to owner",
                  p.get("CallSid"), p.get("SessionStatus") or action)
        call = conn.execute("SELECT id FROM calls WHERE call_sid = ?", (p.get("CallSid", ""),)).fetchone()
        if call:
            calls.finalize_call(conn, call["id"], status="transferred")
        return twiml(say(TROUBLE) + dial_owner(shop, shop["twilio_number"], p.get("CallSid", "")))
