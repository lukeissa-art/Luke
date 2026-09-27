"""Standalone worker, for when the web process runs with RUN_WORKER=false.

python -m scripts.worker
"""

import logging

from app.worker import run_forever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    run_forever()
