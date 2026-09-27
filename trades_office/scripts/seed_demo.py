"""Create a demo shop so you can try the dashboard and a test call right away.

python -m scripts.seed_demo
"""

import json
from datetime import timedelta

from app import db
from app.plans import TRIAL_DAYS


def main() -> int:
    db.init_db()
    now = db.utcnow()
    with db.session() as conn:
        existing = conn.execute("SELECT id FROM shops WHERE name = 'Riverside Plumbing & Heating'").fetchone()
        if existing:
            print(f"Demo shop already exists (id {existing['id']})")
            return existing["id"]
        shop_id = db.insert(conn, "shops", {
            "name": "Riverside Plumbing & Heating",
            "trade": "plumbing",
            "owner_name": "Dave",
            "owner_phone": "+15125550100",
            "twilio_number": "+15125550199",
            "timezone": "America/Chicago",
            "service_area": "Austin, Round Rock, Pflugerville and Cedar Park",
            "services": "Leak repair, drain cleaning, water heaters (tank and tankless), toilets, faucets, "
                        "garbage disposals, sewer line camera inspections, gas line repair",
            "pricing_notes": "$89 service call, waived if the customer goes ahead with the repair. "
                             "After-hours emergency visits are $189.",
            "extra_instructions": "We don't do septic systems or well pumps. Commercial jobs: take a message for Dave.",
            "business_hours": json.dumps({**db.DEFAULT_HOURS, "sat": ["09:00", "13:00"]}),
            "google_review_url": "https://g.page/r/example-review-link",
            "answer_mode": "always",
            "plan": "pro",
            "status": "trial",
            "trial_ends_at": db.iso(now + timedelta(days=TRIAL_DAYS)),
            "created_at": db.iso(now),
        })
    print(f"Created demo shop id {shop_id}")
    return shop_id


if __name__ == "__main__":
    main()
