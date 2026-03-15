from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from trader.alpaca_client import AlpacaAPIError, AlpacaClient
from trader.config import get_settings
from trader.trader_engine import TraderEngine


settings = get_settings()
alpaca = AlpacaClient(settings)
engine = TraderEngine(alpaca, settings)

app = FastAPI(
    title="Alpaca Trader API",
    description="Momentum stock trader with risk controls and Alpaca execution.",
    version="1.0.0",
)


class RunOnceRequest(BaseModel):
    symbols: Optional[List[str]] = None
    force: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict:
    try:
        return engine.get_status()
    except AlpacaAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/signal/{symbol}")
def signal(symbol: str) -> dict:
    try:
        return engine.get_signal(symbol).to_dict()
    except (ValueError, AlpacaAPIError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/run-once")
def run_once(request: RunOnceRequest) -> dict:
    try:
        return engine.run_once(symbols=request.symbols, force=request.force)
    except (ValueError, AlpacaAPIError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
