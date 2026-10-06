"""Website chat for shops: an embeddable chat button their customers use on the shop's own website.

A shop pastes one line on their site:
    <script src="https://<our domain>/widget.js" data-shop="<widget key>" async></script>
That adds a "Chat with us" button which opens /chat/<widget key> in a small panel (an iframe, so
the shop's site styles can't break it). The same page works on its own as a shareable link.
"""

import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from . import calls, db
from .config import get_settings
from .receptionist import Receptionist, greeting, shop_phone
from .streaming import stream_reply, wants_stream

log = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

HEX_COLOR = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?")
MAX_CHARS = 500
MAX_TURNS = 30
LIVE_STATUSES = ("trial", "active")


def snippet(widget_key: str) -> str:
    base = get_settings().public_base_url
    return f'<script src="{base}/widget.js" data-shop="{widget_key}" async></script>'


def chat_link(widget_key: str) -> str:
    return f"{get_settings().public_base_url}/chat/{widget_key}"


@router.get("/widget.js")
def widget_js():
    js = (Path(__file__).parent / "static" / "widget.js").read_text()
    return Response(js, media_type="text/javascript",
                    headers={"Cache-Control": "public, max-age=300", "Access-Control-Allow-Origin": "*"})


@router.get("/chat/{widget_key}", response_class=HTMLResponse)
def chat_page(request: Request, widget_key: str, embed: int = 0, color: str = ""):
    with db.session() as conn:
        shop = db.shop_by_widget_key(conn, widget_key)
    if shop is None:
        raise HTTPException(status_code=404)
    live = shop["status"] in LIVE_STATUSES
    return templates.TemplateResponse(request, "chat.html", {
        "shop": shop, "widget_key": widget_key, "embed": bool(embed), "live": live,
        "greeting": greeting(shop, "web"), "phone": shop_phone(shop),
        "company": get_settings().company_name,
        # Only a plain hex color is accepted, so nothing else can be injected into the page's CSS.
        "accent": color if HEX_COLOR.fullmatch(color or "") else "",
    })


@router.post("/chat/{widget_key}/say")
async def chat_say(widget_key: str, request: Request):
    from .site import _client_ip, _rate_limited

    if _rate_limited("chat:" + _client_ip(request)):
        return JSONResponse({"error": "Too many messages for now. Please call us instead."}, status_code=429)
    body = await request.json()
    if wants_stream(request):
        return stream_reply(lambda on_text: _chat_say(widget_key, body, on_text))
    # The AI call takes seconds: run it off the event loop so the server stays responsive.
    return await run_in_threadpool(_chat_say, widget_key, body)


def _chat_say(widget_key: str, body: dict, on_text=None):
    text = str(body.get("text", "")).strip()[:MAX_CHARS]
    if not text:
        return JSONResponse({"error": "Type a message first."}, status_code=400)
    with db.session() as conn:
        shop = db.shop_by_widget_key(conn, widget_key)
        if shop is None:
            return JSONResponse({"error": "This chat isn't available."}, status_code=404)
        phone = shop_phone(shop)
        if shop["status"] not in LIVE_STATUSES:
            return JSONResponse({"say": f"Our chat is offline right now. Please call us at {phone}.",
                                 "action": "hangup", "chat_id": None})

        call = None
        if body.get("chat_id"):
            call = db.get(conn, "calls", int(body["chat_id"]))
            if call is None or call["shop_id"] != shop["id"] or not call["call_sid"].startswith("web-"):
                call = None
        if call is None:
            if calls.over_cap(conn, shop):
                return JSONResponse({"say": f"Please call us at {phone} and we'll help you right away.",
                                     "action": "hangup", "chat_id": None})
            call = calls.start_call(conn, shop, f"web-{uuid.uuid4().hex[:16]}", "")
        if call["owner_notified"] or call["transcript"].count('"caller"') >= MAX_TURNS:
            return JSONResponse({"say": "This chat has ended. Send a new message to start another.",
                                 "action": "hangup", "chat_id": None})

        turn = Receptionist(conn, shop, call).respond(text, on_text)
        say = turn.say
        if turn.action == "transfer":
            # A website visitor can't be put through: make sure they have the number to call.
            if phone not in say:
                say += f" Please call us at {phone}."
            calls.finalize_call(conn, call["id"], status="transferred")
        elif turn.action == "hangup":
            calls.finalize_call(conn, call["id"])
    return JSONResponse({"say": say, "action": turn.action, "chat_id": call["id"]})
