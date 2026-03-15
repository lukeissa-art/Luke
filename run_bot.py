from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from trader.alpaca_client import AlpacaClient
from trader.config import get_settings
from trader.trader_engine import TraderEngine


def main() -> None:
    settings = get_settings()
    client = AlpacaClient(settings)
    engine = TraderEngine(client, settings)

    while True:
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            result = engine.run_once()
            print(f"[{timestamp}] {json.dumps(result)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep daemon loop alive
            print(
                f"[{timestamp}] "
                f"{json.dumps({'executed': False, 'error': str(exc)})}",
                flush=True,
            )
        time.sleep(settings.poll_interval_seconds)


if __name__ == "__main__":
    main()
