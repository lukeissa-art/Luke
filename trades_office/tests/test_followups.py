from datetime import timedelta

from app import calls, db, followups, scheduling
from tests.conftest import NOW, make_shop


def _call(conn, shop, **values):
    call = calls.start_call(conn, shop, "CA9", "+15125550142", now=NOW)
    db.update(conn, "calls", call["id"], values)
    return db.get(conn, "calls", call["id"])


def test_message_call_notifies_owner_and_queues_text_back(conn, shop):
    call = _call(conn, shop, caller_name="Bo Smith", problem="Clogged drain", outcome="message_taken")
    calls.finalize_call(conn, call["id"], now=NOW)
    owner = conn.execute("SELECT body FROM sms_log WHERE to_phone = ?", (shop["owner_phone"],)).fetchone()
    assert owner["body"].startswith("CALL BACK: Bo Smith")
    f = conn.execute("SELECT * FROM followups").fetchone()
    assert f["kind"] == "unbooked_caller" and f["body"].startswith("Hi Bo")

    # finalize is idempotent (hangup path + Twilio status callback)
    calls.finalize_call(conn, call["id"], now=NOW)
    assert conn.execute("SELECT COUNT(*) FROM followups").fetchone()[0] == 1

    assert followups.dispatch_due(conn, NOW) == 0
    assert followups.dispatch_due(conn, NOW + timedelta(minutes=20)) == 1
    assert db.get(conn, "followups", f["id"])["status"] == "sent"


def test_text_back_skipped_if_caller_booked_meanwhile(conn, shop):
    call = _call(conn, shop, caller_name="Bo", problem="Clog", outcome="message_taken")
    calls.finalize_call(conn, call["id"], now=NOW)
    slot = scheduling.open_slots(conn, shop, now=NOW)[0]
    scheduling.book(conn, shop, start_iso=slot["start"], customer_name="Bo", customer_phone="+15125550142",
                    address="1 Elm", problem="Clog", now=NOW + timedelta(minutes=5))
    assert followups.dispatch_due(conn, NOW + timedelta(hours=1)) == 0
    assert conn.execute("SELECT status FROM followups").fetchone()["status"] == "skipped"


def test_quiet_hours(shop):
    late = db.parse_iso("2026-10-01T03:30:00+00:00")  # 10:30pm Chicago
    pushed = followups.next_allowed_time(shop, late)
    assert (pushed.hour, pushed.day) == (8, 1)


def test_job_done_queues_review_request(conn, shop):
    slot = scheduling.open_slots(conn, shop, now=NOW)[0]
    appt_id = scheduling.book(conn, shop, start_iso=slot["start"], customer_name="Ann Lee",
                              customer_phone="+15125550142", address="1 Elm", problem="Leak", now=NOW)
    calls.complete_appointment(conn, appt_id, now=NOW)
    f = conn.execute("SELECT * FROM followups").fetchone()
    assert f["kind"] == "review_request" and "g.page" in f["body"]


def test_plus_plan_quote_followups_and_maintenance(conn):
    shop = make_shop(conn, plan="plus", trade="hvac", twilio_number="+15125550197")
    qid = db.insert(conn, "quotes", {"shop_id": shop["id"], "customer_name": "Cy", "customer_phone": "+15125550143",
                                     "description": "new AC unit", "amount": 6400, "created_at": db.iso(NOW)})
    assert len(followups.schedule_quote_followups(conn, shop, db.get(conn, "quotes", qid), now=NOW)) == 2
    db.update(conn, "quotes", qid, {"status": "won"})
    assert followups.dispatch_due(conn, NOW + timedelta(days=30)) == 0  # won quotes aren't nagged

    slot = scheduling.open_slots(conn, shop, now=NOW)[0]
    appt_id = scheduling.book(conn, shop, start_iso=slot["start"], customer_name="Cy",
                              customer_phone="+15125550143", address="1 Elm", problem="AC", now=NOW)
    calls.complete_appointment(conn, appt_id, now=NOW)
    kinds = {r["kind"] for r in conn.execute("SELECT kind FROM followups WHERE status = 'pending'")}
    assert kinds == {"review_request", "maintenance"}


def test_opted_out_customer_gets_nothing(conn, shop):
    conn.execute("INSERT INTO opt_outs VALUES (?, ?, ?)", (shop["id"], "+15125550142", db.iso(NOW)))
    call = _call(conn, shop, caller_name="Bo", problem="Clog", outcome="message_taken")
    calls.finalize_call(conn, call["id"], now=NOW)
    assert followups.dispatch_due(conn, NOW + timedelta(hours=1)) == 0
