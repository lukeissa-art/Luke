"""Give a shop its own assistant phone number in one click.

Buys a local number on our Twilio account (or takes one already on it), points its voice and
text webhooks at this app, and returns it so it can be saved as the shop's assistant line.
"""

import logging
import re
from urllib.parse import urlencode
from xml.sax.saxutils import escape

from . import sms
from .config import get_settings

log = logging.getLogger(__name__)


class PhoneNumberError(Exception):
    """A problem worth showing to whoever clicked the button."""


# Twilio-hosted "echo" Twimlet: returns the TwiML we give it. Used as the number's fallback URL,
# so calls still reach the owner when this app is down (outage, bad deploy, out of AI credit).
TWIMLET_ECHO = "https://twimlets.com/echo"


def fallback_url(shop) -> str:
    """Where Twilio sends a call if this app doesn't answer: straight to the owner's phone."""
    owner = (shop["owner_phone"] or "").strip()
    if not owner.startswith("+"):
        return ""
    twiml = (f'<Response><Say voice="{escape(get_settings().tts_voice, {chr(34): "&quot;"})}">'
             f'Thanks for calling {escape(shop["name"])}. Connecting you now.</Say>'
             f'<Dial timeout="25">{escape(owner)}</Dial></Response>')
    return f"{TWIMLET_ECHO}?{urlencode({'Twiml': twiml})}"


def webhooks(shop=None) -> dict[str, str]:
    base = get_settings().public_base_url
    hooks = {
        "voice_url": f"{base}/voice/incoming",
        "voice_method": "POST",
        "status_callback": f"{base}/voice/status",
        "status_callback_method": "POST",
        "sms_url": f"{base}/sms/incoming",
        "sms_method": "POST",
    }
    if shop is not None and fallback_url(shop):
        hooks.update(voice_fallback_url=fallback_url(shop), voice_fallback_method="GET")
    return hooks


def _client():
    if not get_settings().twilio_enabled:
        raise PhoneNumberError("Twilio isn't connected yet. Add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN "
                               "to the server's environment variables, then try again.")
    return sms._twilio()


def _twilio_message(exc: Exception) -> str:
    return getattr(exc, "msg", None) or str(exc) or exc.__class__.__name__


def buy_number(area_code: str, shop) -> str:
    """Buy a local voice + SMS number in the area code and connect it. Returns it in E.164 form."""
    area_code = re.sub(r"\D", "", area_code or "")
    if len(area_code) != 3:
        raise PhoneNumberError("Enter a 3-digit area code, like 512.")
    client = _client()
    try:
        found = client.available_phone_numbers("US").local.list(
            area_code=area_code, voice_enabled=True, sms_enabled=True, limit=1)
    except Exception as exc:
        log.exception("Twilio number search failed")
        raise PhoneNumberError(f"Twilio couldn't search for numbers: {_twilio_message(exc)}") from exc
    if not found:
        raise PhoneNumberError(f"No numbers are available in area code {area_code} right now. "
                               "Try a nearby area code.")
    try:
        bought = client.incoming_phone_numbers.create(
            phone_number=found[0].phone_number, friendly_name=f"{shop['name']} assistant"[:64],
            **webhooks(shop))
    except Exception as exc:
        log.exception("Twilio number purchase failed")
        raise PhoneNumberError(f"Twilio couldn't buy the number: {_twilio_message(exc)}") from exc
    log.info("Bought %s for %s", bought.phone_number, shop["name"])
    return bought.phone_number


def _find_owned(client, number: str):
    try:
        owned = client.incoming_phone_numbers.list(phone_number=number, limit=1)
    except Exception as exc:
        log.exception("Twilio number lookup failed")
        raise PhoneNumberError(f"Twilio couldn't look up the number: {_twilio_message(exc)}") from exc
    return owned[0] if owned else None


def connect_number(number: str, shop) -> str:
    """Point a number already on our Twilio account at this app. `number` must be E.164."""
    client = _client()
    owned = _find_owned(client, number)
    if owned is None:
        raise PhoneNumberError(f"{number} isn't on your Twilio account. Buy it (or port it in) on Twilio "
                               "first, or use \"Get a new number\" instead.")
    try:
        client.incoming_phone_numbers(owned.sid).update(friendly_name=f"{shop['name']} assistant"[:64],
                                                        **webhooks(shop))
    except Exception as exc:
        log.exception("Twilio number update failed")
        raise PhoneNumberError(f"Twilio couldn't update the number: {_twilio_message(exc)}") from exc
    return owned.phone_number


def release_number(number: str) -> None:
    """Give the number back to Twilio so it stops costing money."""
    client = _client()
    owned = _find_owned(client, number)
    if owned is None:
        return  # already gone from the account
    try:
        client.incoming_phone_numbers(owned.sid).delete()
    except Exception as exc:
        log.exception("Twilio number release failed")
        raise PhoneNumberError(f"Twilio couldn't release the number: {_twilio_message(exc)}") from exc
