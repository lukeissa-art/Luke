"""Background jobs: due follow-up texts every minute, weekly reports Monday morning.

Runs inside the web process by default (RUN_WORKER=true) so the whole product deploys as a
single service sharing one SQLite file. Run only one copy: one web instance, one worker.
"""

import logging
import threading
from datetime import datetime

from . import calls, db, followups, reports

log = logging.getLogger("worker")

REPORT_WEEKDAY = 0  # Monday
REPORT_HOUR_UTC = 13  # ~8am Central


def tick(now: datetime, last_report_day=None):
    """One pass of the job loop. Returns the date weekly reports last went out."""
    with db.session() as conn:
        closed = calls.close_stale_web_chats(conn, now)
        if closed:
            log.info("Closed %d finished website chats", closed)
        sent = followups.dispatch_due(conn, now)
        if sent:
            log.info("Sent %d follow-up texts", sent)
        if now.weekday() == REPORT_WEEKDAY and now.hour == REPORT_HOUR_UTC and last_report_day != now.date():
            log.info("Sent %d weekly reports", reports.send_weekly_reports(conn, now))
            return now.date()
    return last_report_day


def run_forever(interval: int = 60, stop: threading.Event | None = None) -> None:
    db.init_db()
    last_report_day = None
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            last_report_day = tick(db.utcnow(), last_report_day)
        except Exception:
            log.exception("Worker tick failed")
        stop.wait(interval)


def start_background() -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=run_forever, kwargs={"stop": stop}, daemon=True, name="worker").start()
    log.info("Background worker started")
    return stop
