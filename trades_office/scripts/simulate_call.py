"""Talk to a shop's AI receptionist from the terminal (needs ANTHROPIC_API_KEY or `ant auth login`).

python -m scripts.simulate_call [shop_id]
"""

import sys
import uuid

from app import calls, db
from app.receptionist import Receptionist, greeting


def main() -> None:
    db.init_db()
    with db.session() as conn:
        shop_id = int(sys.argv[1]) if len(sys.argv) > 1 else conn.execute("SELECT MIN(id) FROM shops").fetchone()[0]
        if shop_id is None:
            sys.exit("No shops yet. Run: python -m scripts.seed_demo")
        shop = db.get(conn, "shops", shop_id)
        call = calls.start_call(conn, shop, f"demo-{uuid.uuid4().hex[:12]}", "+15125550142")
        conn.commit()
        print(f"\n[{shop['name']}] Assistant: {greeting(shop)}")
        while True:
            try:
                text = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            call = db.get(conn, "calls", call["id"])
            turn = Receptionist(conn, shop, call).respond(text)
            conn.commit()
            print(f"Assistant: {turn.say}")
            if turn.action != "continue":
                print(f"[{'transferring to ' + shop['owner_phone'] if turn.action == 'transfer' else 'call ended'}]")
                break
        calls.finalize_call(conn, call["id"])
        call = db.get(conn, "calls", call["id"])
        print(f"\nOutcome: {call['outcome']}\nOwner summary: {call['summary'] or call['problem']}")


if __name__ == "__main__":
    main()
