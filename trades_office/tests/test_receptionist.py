import anthropic
import httpx

from app import calls, db, scheduling
from app.receptionist import Receptionist
from tests.conftest import NOW, FakeClaude, make_shop, reply, text, tool


def start(conn, shop):
    call = calls.start_call(conn, shop, "CA123", "+15125550142", now=NOW)
    conn.commit()
    return call


def respond(conn, shop, call_id, fake, utterance):
    call = db.get(conn, "calls", call_id)
    return Receptionist(conn, shop, call, client=fake, now=NOW).respond(utterance)


def test_full_booking_call(conn, shop):
    call = start(conn, shop)
    slot = scheduling.open_slots(conn, shop, now=NOW)[0]
    fake = FakeClaude([
        reply(tool("check_availability", {"preferred_date": ""})),
        reply(text("I'm sorry about the leak. I can have someone out today between 12 and 2. Does that work?")),
        reply(tool("book_appointment", {
            "slot_start": slot["start"], "customer_name": "Ann Lee", "callback_phone": "512-555-0142",
            "address": "12 Oak St, Austin", "problem_summary": "Water heater leaking in garage"}, "tu_2")),
        reply(text("You're all set for today between 12 and 2. Goodbye!"),
              tool("end_call", {"outcome": "booked", "summary": "Water heater leak, booked today 12-2."}, "tu_3")),
    ])

    t1 = respond(conn, shop, call["id"], fake, "My water heater is leaking")
    assert t1.action == "continue" and "12 and 2" in t1.say

    t2 = respond(conn, shop, call["id"], fake, "Yes, Ann Lee, 12 Oak Street, this number is fine")
    assert t2.action == "hangup" and "all set" in t2.say

    appt = conn.execute("SELECT * FROM appointments").fetchone()
    assert appt["customer_name"] == "Ann Lee" and appt["customer_phone"] == "+15125550142"
    row = db.get(conn, "calls", call["id"])
    assert row["outcome"] == "booked" and row["caller_name"] == "Ann Lee"
    assert "booked" in conn.execute("SELECT body FROM sms_log").fetchone()["body"]

    # History persists across webhooks: the second request replays the first turn plus tool results.
    second_turn = fake.requests[2]["messages"]
    assert second_turn[0]["content"].startswith("[Call started Wednesday September 30 2026, 10:00 AM")
    assert second_turn[1]["content"][0]["type"] == "tool_use"
    assert second_turn[2]["content"][0]["type"] == "tool_result"

    req = fake.requests[0]
    assert req["model"] == "claude-opus-5"
    assert req["output_config"] == {"effort": "low"}
    assert req["fallbacks"] == "default"
    assert {t["name"] for t in req["tools"]} >= {"check_availability", "book_appointment", "end_call"}
    assert all(t["strict"] for t in req["tools"])

    calls.finalize_call(conn, call["id"], now=NOW)
    summary = conn.execute("SELECT body FROM sms_log WHERE to_phone = ?", (shop["owner_phone"],)).fetchone()
    assert summary["body"].startswith("BOOKED: Ann Lee")
    assert conn.execute("SELECT COUNT(*) FROM followups").fetchone()[0] == 0


def test_gas_leak_transfers_without_asking_the_model(conn, shop):
    call = start(conn, shop)
    fake = FakeClaude([])  # any model call would fail the test
    turn = respond(conn, shop, call["id"], fake, "I smell gas in my basement")
    assert turn.action == "transfer"
    assert "leave the building" in turn.say
    assert db.get(conn, "calls", call["id"])["is_emergency"] == 1
    urgent = conn.execute("SELECT body FROM sms_log WHERE to_phone = ?", (shop["owner_phone"],)).fetchone()
    assert urgent["body"].startswith("URGENT")


def test_urgent_keyword_is_flagged_and_passed_to_model(conn, shop):
    call = start(conn, shop)
    fake = FakeClaude([reply(text("That sounds urgent. Would you like me to connect you to Dave right now?"))])
    turn = respond(conn, shop, call["id"], fake, "a pipe burst and water is everywhere, it's flooding")
    assert turn.action == "continue"
    assert "urgent keyword detected" in fake.requests[0]["messages"][0]["content"]
    assert db.get(conn, "calls", call["id"])["is_emergency"] == 1


def test_api_failure_transfers_to_owner(conn, shop):
    call = start(conn, shop)
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    fake = FakeClaude([err])
    turn = respond(conn, shop, call["id"], fake, "hello?")
    assert turn.action == "transfer"


def test_refusal_takes_a_message(conn, shop):
    call = start(conn, shop)
    fake = FakeClaude([reply(stop="refusal")])
    turn = respond(conn, shop, call["id"], fake, "something odd")
    assert turn.action == "hangup" and "call you back" in turn.say
    assert db.get(conn, "calls", call["id"])["outcome"] == "message_taken"


def test_starter_plan_cannot_book(conn):
    shop = make_shop(conn, plan="starter", twilio_number="+15125550198")
    call = start(conn, shop)
    fake = FakeClaude([reply(text("I'll have someone call you back."))])
    respond(conn, shop, call["id"], fake, "can I get someone out tomorrow?")
    names = {t["name"] for t in fake.requests[0]["tools"]}
    assert "book_appointment" not in names and "record_caller_details" in names
    assert "cannot book appointments" in fake.requests[0]["system"]


def test_bad_slot_is_reported_back_to_model(conn, shop):
    call = start(conn, shop)
    fake = FakeClaude([
        reply(tool("book_appointment", {"slot_start": "2026-09-30T03:00:00+00:00", "customer_name": "A",
                                        "callback_phone": "", "address": "x", "problem_summary": "y"})),
        reply(text("Sorry, that time just filled up. Let me check again.")),
    ])
    respond(conn, shop, call["id"], fake, "book me at 10pm")
    result = fake.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "check_availability" in result["content"]


def test_missing_credentials_still_transfer(conn, shop):
    call = start(conn, shop)
    fake = FakeClaude([TypeError("Could not resolve authentication method")])
    assert respond(conn, shop, call["id"], fake, "hello?").action == "transfer"
