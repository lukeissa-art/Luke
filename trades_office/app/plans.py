"""Pricing plans from the business plan. Call caps protect margins; features gate the tiers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    key: str
    name: str
    monthly_price: int
    monthly_call_cap: int
    answers_24_7: bool
    booking: bool
    followups: bool
    review_requests: bool
    quote_followups: bool
    maintenance_reminders: bool
    multi_location: bool


PLANS: dict[str, Plan] = {
    "starter": Plan("starter", "Starter", 249, 150, False, False, False, False, False, False, False),
    "pro": Plan("pro", "Pro", 449, 400, True, True, True, True, False, False, False),
    "plus": Plan("plus", "Plus", 799, 1000, True, True, True, True, True, True, True),
}

SETUP_FEE = 500
TRIAL_DAYS = 14


def get_plan(key: str) -> Plan:
    return PLANS.get(key, PLANS["pro"])
