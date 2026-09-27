import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

from . import admin, billing, db, telephony
from .config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title=get_settings().company_name, lifespan=lifespan)
app.include_router(telephony.router)
app.include_router(billing.router)
app.include_router(admin.router, prefix="/admin")


@app.get("/")
def root():
    return RedirectResponse("/admin/")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/billing/thanks", response_class=HTMLResponse)
def thanks():
    return "<h2>You're all set. Thanks for your business!</h2>"


@app.get("/billing/cancelled", response_class=HTMLResponse)
def cancelled():
    return "<h2>No charge was made. Call or text us anytime with questions.</h2>"
