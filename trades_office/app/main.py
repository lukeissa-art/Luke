import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from . import admin, billing, db, sales, site, telephony, widget, worker
from .config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    if get_settings().seed_demo:
        from scripts import seed_demo

        seed_demo.main()
    stop = None
    if get_settings().run_worker:
        stop = worker.start_background()
        # Re-point shops' numbers at this address in the background (doesn't delay startup).
        from . import phone_numbers

        threading.Thread(target=phone_numbers.sync_all, name="webhook-sync", daemon=True).start()
    yield
    if stop:
        stop.set()


app = FastAPI(title=get_settings().company_name, lifespan=lifespan)
app.include_router(site.router)
app.include_router(sales.router)
app.include_router(widget.router)
app.include_router(telephony.router)
app.include_router(billing.router)
app.include_router(admin.router, prefix="/admin")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/billing/thanks", response_class=HTMLResponse)
def thanks():
    return "<h2>You're all set. Thanks for your business!</h2>"


@app.get("/billing/cancelled", response_class=HTMLResponse)
def cancelled():
    return "<h2>No charge was made. Call or text us anytime with questions.</h2>"
