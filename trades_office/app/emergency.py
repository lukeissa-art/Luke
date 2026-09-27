"""Hard emergency rules. These run on every caller utterance *before* the AI sees it,
so a gas leak is never left to model judgment."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EmergencyMatch:
    level: str  # "life_safety" or "urgent"
    keyword: str
    caller_instructions: str


# Life-safety: tell the caller what to do *first*, then transfer to the owner.
LIFE_SAFETY = {
    r"\bgas (smell|leak)|smell(s|ing)? (of |like )?gas|rotten eggs?\b": (
        "If you smell gas, please leave the building now, don't flip any switches, "
        "and call 911 or your gas company from outside."
    ),
    r"carbon monoxide|\bco (alarm|detector)\b|co alarm": (
        "If your carbon monoxide alarm is going off, get everyone outside to fresh air now and call 911."
    ),
    r"\b(sparks?|sparking|arcing)\b|smoke coming|burning smell|electrical fire|\bon fire\b": (
        "If you see sparks, smoke or smell burning, stay away from it and call 911 if there is any fire. "
        "If it's safe, switch off the main breaker."
    ),
}

# Urgent: property damage or vulnerable people. Transfer to the owner, no safety script needed.
URGENT = [
    r"\bflood(ing|ed)?\b",
    r"burst (pipe|line)",
    r"water (is )?(pouring|everywhere|gushing)",
    r"sewage|sewer (backup|backing up)",
    r"no heat",
    r"\bno (ac|a/c|air conditioning|cooling)\b",
    r"power (is )?out|no power",
]


def check(text: str, extra_keywords: str = "") -> EmergencyMatch | None:
    lowered = text.lower()
    for pattern, instructions in LIFE_SAFETY.items():
        m = re.search(pattern, lowered)
        if m:
            return EmergencyMatch("life_safety", m.group(0), instructions)
    extras = [re.escape(k.strip().lower()) for k in extra_keywords.split(",") if k.strip()]
    for pattern in URGENT + extras:
        m = re.search(pattern, lowered)
        if m:
            return EmergencyMatch("urgent", m.group(0), "")
    return None
