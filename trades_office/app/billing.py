"""Stripe subscriptions: a checkout link per shop, and a webhook that keeps shop status in sync."""

import logging

from fastapi import APIRouter, HTTPException, Request

from . import db
from .config import get_settings
from .plans import SETUP_FEE

log = logging.getLogger(__name__)
router = APIRouter()


def _stripe():
    import stripe

    stripe.api_key = get_settings().stripe_api_key
    return stripe


def price_id(plan: str) -> str:
    s = get_settings()
    return {"starter": s.stripe_price_starter, "pro": s.stripe_price_pro, "plus": s.stripe_price_plus}[plan]


def checkout_url(shop, *, waive_setup_fee: bool) -> str:
    """Create a Stripe Checkout link to text or email to the owner at the end of their trial."""
    stripe = _stripe()
    line_items = [{"price": price_id(shop["plan"]), "quantity": 1}]
    if not waive_setup_fee:
        line_items.append({
            "price_data": {"currency": "usd", "unit_amount": SETUP_FEE * 100,
                           "product_data": {"name": "One-time setup"}},
            "quantity": 1,
        })
    base = get_settings().public_base_url
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=line_items,
        client_reference_id=str(shop["id"]),
        success_url=f"{base}/billing/thanks",
        cancel_url=f"{base}/billing/cancelled",
    )
    return session.url


@router.post("/billing/webhook")
async def stripe_webhook(request: Request):
    s = get_settings()
    payload = await request.body()
    try:
        event = _stripe().Webhook.construct_event(
            payload, request.headers.get("Stripe-Signature", ""), s.stripe_webhook_secret
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature")
    apply_event(event["type"], event["data"]["object"])
    return {"ok": True}


def apply_event(event_type: str, obj) -> None:
    with db.session() as conn:
        if event_type == "checkout.session.completed" and obj.get("client_reference_id"):
            db.update(conn, "shops", int(obj["client_reference_id"]), {
                "status": "active",
                "stripe_customer_id": obj.get("customer"),
                "stripe_subscription_id": obj.get("subscription"),
            })
            return
        sub_id = obj.get("id") if event_type.startswith("customer.subscription") else obj.get("subscription")
        if not sub_id:
            return
        shop = conn.execute("SELECT * FROM shops WHERE stripe_subscription_id = ?", (sub_id,)).fetchone()
        if shop is None:
            return
        if event_type == "customer.subscription.deleted":
            db.update(conn, "shops", shop["id"], {"status": "cancelled"})
        elif event_type == "invoice.payment_failed":
            log.warning("Payment failed for shop %s", shop["id"])
        elif event_type == "customer.subscription.updated" and obj.get("status") in ("active", "trialing"):
            if shop["status"] != "paused":
                db.update(conn, "shops", shop["id"], {"status": "active"})
