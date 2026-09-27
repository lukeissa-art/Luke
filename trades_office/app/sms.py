"""Outbound texts. Without Twilio credentials, messages are logged instead of sent
so the whole product can be demoed locally."""

import logging
import sqlite3

from . import db
from .config import get_settings

log = logging.getLogger(__name__)

_client = None


def _twilio():
    global _client
    if _client is None:
        from twilio.rest import Client

        s = get_settings()
        _client = Client(s.twilio_account_sid, s.twilio_auth_token)
    return _client


def is_opted_out(conn: sqlite3.Connection, shop_id: int, phone: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM opt_outs WHERE shop_id = ? AND phone = ?", (shop_id, phone)
    ).fetchone()
    return row is not None


def send(
    conn: sqlite3.Connection,
    shop: sqlite3.Row,
    to: str,
    body: str,
    *,
    to_owner: bool = False,
) -> bool:
    """Send a text from the shop's number. Customer texts respect STOP opt-outs."""
    if not to:
        return False
    if not to_owner and is_opted_out(conn, shop["id"], to):
        log.info("Skipping text to opted-out %s", to)
        return False
    from_number = shop["twilio_number"] or ""
    if get_settings().twilio_enabled and from_number:
        try:
            _twilio().messages.create(to=to, from_=from_number, body=body)
        except Exception:
            log.exception("SMS to %s failed", to)
            return False
    else:
        log.info("[dry-run SMS] %s -> %s: %s", from_number, to, body)
    db.insert(
        conn,
        "sms_log",
        {
            "shop_id": shop["id"],
            "direction": "outbound",
            "to_phone": to,
            "from_phone": from_number,
            "body": body,
            "created_at": db.iso(db.utcnow()),
        },
    )
    return True
