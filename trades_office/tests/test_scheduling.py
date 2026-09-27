import pytest

from app import db, scheduling
from tests.conftest import NOW


def test_open_slots_respect_hours_and_lead_time(conn, shop):
    slots = scheduling.open_slots(conn, shop, now=NOW, limit=4)
    # 10:00 local now + 2h lead -> first window is 1pm (8-10, 10-12 too soon; 12-2 starts in 2h exactly)
    assert slots[0]["label"].startswith("Wednesday September 30, 12 PM")
    assert len(slots) == 4
    assert slots[2]["label"].startswith("Thursday October 1, 8 AM")


def test_booked_window_is_removed(conn, shop):
    first = scheduling.open_slots(conn, shop, now=NOW)[0]
    scheduling.book(conn, shop, start_iso=first["start"], customer_name="Ann", customer_phone="+15125550142",
                    address="1 Main St", problem="leak", now=NOW)
    assert scheduling.open_slots(conn, shop, now=NOW)[0]["start"] != first["start"]
    with pytest.raises(scheduling.SlotUnavailable):
        scheduling.book(conn, shop, start_iso=first["start"], customer_name="Bob", customer_phone="+1",
                        address="2 Main St", problem="clog", now=NOW)


def test_weekend_closed_by_default(conn, shop):
    saturday = db.parse_iso("2026-10-03T15:00:00+00:00")
    assert not scheduling.is_open_now(shop, saturday)
    assert scheduling.is_open_now(shop, NOW)
