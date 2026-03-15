from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

from trader.config import Settings

# Symbols containing "/" are treated as crypto (e.g. "BTC/USD", "ETH/USD")
CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD", "LINK/USD", "LTC/USD"}


def is_crypto(symbol: str) -> bool:
    return "/" in symbol or symbol.upper() in CRYPTO_SYMBOLS


class AlpacaAPIError(RuntimeError):
    """Raised when Alpaca responds with an error."""


class AlpacaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": settings.alpaca_api_key,
                "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
                "Content-Type": "application/json",
            }
        )

    def _request(
        self,
        method: str,
        base_url: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        allow_404: bool = False,
    ) -> Any:
        url = f"{base_url.rstrip('/')}{path}"
        response = self.session.request(
            method,
            url,
            params=params,
            json=json_body,
            timeout=20,
        )

        if allow_404 and response.status_code == 404:
            return None

        if response.status_code == 204:
            return {}

        if not response.ok:
            try:
                payload = response.json()
                message = payload.get("message") or str(payload)
            except ValueError:
                message = response.text
            raise AlpacaAPIError(
                f"Alpaca request failed ({response.status_code}) at {path}: {message}"
            )

        try:
            return response.json()
        except ValueError:
            return {}

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", self.settings.alpaca_base_url, "/v2/account")

    def get_clock(self) -> dict[str, Any]:
        return self._request("GET", self.settings.alpaca_base_url, "/v2/clock")

    def is_market_open(self) -> bool:
        clock = self.get_clock()
        return bool(clock.get("is_open"))

    def list_positions(self) -> list[dict[str, Any]]:
        return self._request("GET", self.settings.alpaca_base_url, "/v2/positions")

    def get_position(self, symbol: str) -> Optional[dict[str, Any]]:
        return self._request(
            "GET",
            self.settings.alpaca_base_url,
            f"/v2/positions/{symbol.upper()}",
            allow_404=True,
        )

    def get_asset(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get asset info. Handles both stocks and crypto."""
        if is_crypto(symbol):
            # Crypto assets use a different endpoint
            return self._request(
                "GET",
                self.settings.alpaca_base_url,
                f"/v2/assets/{symbol.upper()}",
                allow_404=True,
            )
        return self._request(
            "GET",
            self.settings.alpaca_base_url,
            f"/v2/assets/{symbol.upper()}",
            params={"asset_class": "us_equity"},
            allow_404=True,
        )

    def get_bars(self, symbol: str, timeframe: str, limit: int) -> list[dict[str, Any]]:
        """Get price bars. Routes to crypto or stock endpoint automatically."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=120)

        if is_crypto(symbol):
            payload = self._request(
                "GET",
                self.settings.alpaca_data_url,
                f"/v1beta3/crypto/us/bars",
                params={
                    "symbols": symbol.upper(),
                    "timeframe": timeframe,
                    "limit": limit,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )
            # Crypto bars are nested under the symbol key
            bars_by_symbol = payload.get("bars", {})
            return bars_by_symbol.get(symbol.upper(), [])

        # Stock bars
        payload = self._request(
            "GET",
            self.settings.alpaca_data_url,
            f"/v2/stocks/{symbol.upper()}/bars",
            params={
                "timeframe": timeframe,
                "limit": limit,
                "adjustment": "raw",
                "feed": "iex",
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
        return payload.get("bars", [])

    def get_latest_trade_price(self, symbol: str) -> float:
        """Get latest price. Routes to crypto or stock endpoint automatically."""
        if is_crypto(symbol):
            payload = self._request(
                "GET",
                self.settings.alpaca_data_url,
                f"/v1beta3/crypto/us/latest/trades",
                params={"symbols": symbol.upper()},
            )
            trades = payload.get("trades", {})
            trade = trades.get(symbol.upper(), {})
            price = trade.get("p")
        else:
            payload = self._request(
                "GET",
                self.settings.alpaca_data_url,
                f"/v2/stocks/{symbol.upper()}/trades/latest",
                params={"feed": "iex"},
            )
            trade = payload.get("trade") or {}
            price = trade.get("p")

        if price is None:
            raise AlpacaAPIError(f"Missing latest trade price for {symbol}")
        return float(price)

    def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            self.settings.alpaca_base_url,
            "/v2/orders",
            json_body=order,
        )

    def close_position(self, symbol: str) -> Optional[dict[str, Any]]:
        return self._request(
            "DELETE",
            self.settings.alpaca_base_url,
            f"/v2/positions/{symbol.upper()}",
            allow_404=True,
        )
