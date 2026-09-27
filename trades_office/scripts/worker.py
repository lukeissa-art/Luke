"""Background worker: sends due follow-up texts every minute and weekly reports on Monday morning.

Run alongside the web server:  python -m scripts.worker
"""

import logging
import time

from app import db, followups, reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")

REPORT_WEEKDAY = 0  # Monday
REPORT_HOUR_UTC = 13  # ~8am Central


def run_forever(interval: int = 60) -> None:
    db.init_db()
    last_report_day = None
    while True:
        now = db.utcnow()
        with db.session() as conn:
            sent = followups.dispatch_due(conn, now)
            if sent:
                log.info("Sent %d follow-up texts", sent)
            if now.weekday() == REPORT_WEEKDAY and now.hour == REPORT_HOUR_UTC and last_report_day != now.date():
                log.info("Sent %d weekly reports", reports.send_weekly_reports(conn, now))
                last_report_day = now.date()
        time.sleep(interval)


if __name__ == "__main__":
    run_forever()
