"""Weekly owner report: proof the product pays for itself."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from . import db, scheduling, sms
from .plans import get_plan


@dataclass
class ShopStats:
    calls: int
    after_hours: int
    emergencies: int
    booked: int
    messages: int
    followups_sent: int
    reviews_requested: int
    est_revenue: int
    plan_price: int

    @property
    def roi_multiple(self) -> float:
        weekly_cost = self.plan_price * 12 / 52
        return self.est_revenue / weekly_cost if weekly_cost else 0.0


def stats(conn: sqlite3.Connection, shop: sqlite3.Row, start: datetime, end: datetime) -> ShopStats:
    s, e = db.iso(start), db.iso(end)
    calls = conn.execute(
        "SELECT * FROM calls WHERE shop_id = ? AND started_at >= ? AND started_at < ? "
        "AND call_sid NOT LIKE 'demo-%'",
        (shop["id"], s, e),
    ).fetchall()
    after_hours = sum(1 for c in calls if not scheduling.is_open_now(shop, db.parse_iso(c["started_at"])))
    booked = conn.execute(
        "SELECT COUNT(*) FROM appointments WHERE shop_id = ? AND created_at >= ? AND created_at < ? "
        "AND status != 'cancelled'",
        (shop["id"], s, e),
    ).fetchone()[0]
    sent = dict(
        conn.execute(
            "SELECT kind, COUNT(*) FROM followups WHERE shop_id = ? AND status = 'sent' "
            "AND sent_at >= ? AND sent_at < ? GROUP BY kind",
            (shop["id"], s, e),
        ).fetchall()
    )
    return ShopStats(
        calls=len(calls),
        after_hours=after_hours,
        emergencies=sum(c["is_emergency"] for c in calls),
        booked=booked,
        messages=sum(1 for c in calls if c["outcome"] == "message_taken"),
        followups_sent=sum(sent.values()),
        reviews_requested=sent.get("review_request", 0),
        est_revenue=booked * shop["avg_job_value"],
        plan_price=get_plan(shop["plan"]).monthly_price,
    )


def weekly_text(shop: sqlite3.Row, st: ShopStats) -> str:
    return (
        f"{shop['name']} weekly report: {st.calls} calls answered ({st.after_hours} after hours, "
        f"{st.emergencies} urgent). {st.booked} jobs booked, about ${st.est_revenue:,} in work. "
        f"{st.messages} callbacks for you, {st.followups_sent} follow-up texts sent, "
        f"{st.reviews_requested} review requests."
    )


def send_weekly_reports(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    now = now or db.utcnow()
    count = 0
    for shop in conn.execute("SELECT * FROM shops WHERE status IN ('trial', 'active')").fetchall():
        st = stats(conn, shop, now - timedelta(days=7), now)
        if sms.send(conn, shop, shop["owner_phone"], weekly_text(shop, st), to_owner=True):
            count += 1
    return count
