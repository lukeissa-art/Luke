from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

from trader.alpaca_client import AlpacaAPIError, AlpacaClient, is_crypto
from trader.config import Settings
from trader.strategy import Signal, generate_signal


class TraderEngine:
    def __init__(self, client: AlpacaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.last_trade_at: dict[str, datetime] = {}

    def _format_qty(self, quantity: float, crypto: bool = False) -> str:
        if crypto:
            # Crypto supports fractional quantities up to 8 decimal places
            return f"{quantity:.8f}".rstrip("0").rstrip(".")
        if quantity == int(quantity):
            return str(int(quantity))
        return f"{quantity:.6f}".rstrip("0").rstrip(".")

    def _in_cooldown(self, symbol: str) -> bool:
        last = self.last_trade_at.get(symbol.upper())
        if not last:
            return False
        return datetime.now(timezone.utc) - last < timedelta(
            minutes=self.settings.cooldown_minutes
        )

    def _mark_traded(self, symbol: str) -> None:
        self.last_trade_at[symbol.upper()] = datetime.now(timezone.utc)

    def _position_qty(self, position: dict[str, Any]) -> float:
        raw_qty = position.get("qty")
        if raw_qty is None:
            return 0.0
        return abs(float(raw_qty))

    def _compute_buy_qty(self, equity: float, cash: float, price: float, crypto: bool = False) -> float:
        if price <= 0:
            return 0.0
        budget = min(
            equity * self.settings.max_notional_per_position,
            cash * 0.95,
        )
        if crypto:
            # Crypto supports fractional quantities
            qty = budget / price
            return max(round(qty, 8), 0.0)
        else:
            # Stocks require whole shares
            qty = int(budget / price)
            return float(max(qty, 0))

    def get_status(self) -> dict[str, Any]:
        account = self.client.get_account()
        positions = self.client.list_positions()
        return {
            "dry_run": self.settings.dry_run,
            "symbols": self.settings.symbols_list,
            "equity": float(account.get("equity", 0)),
            "cash": float(account.get("cash", 0)),
            "buying_power": float(account.get("buying_power", 0)),
            "position_count": len(positions),
            "positions": positions,
            "last_trade_at": {
                symbol: timestamp.isoformat()
                for symbol, timestamp in self.last_trade_at.items()
            },
        }

    def get_signal(self, symbol: str) -> Signal:
        bars = self.client.get_bars(
            symbol=symbol,
            timeframe=self.settings.bar_timeframe,
            limit=self.settings.lookback_bars,
        )
        closes = [float(bar["c"]) for bar in bars if "c" in bar]
        return generate_signal(
            symbol=symbol,
            closes=closes,
            fast_period=self.settings.fast_ema,
            slow_period=self.settings.slow_ema,
            rsi_period=self.settings.rsi_period,
            rsi_overbought=self.settings.rsi_overbought,
            rsi_oversold=self.settings.rsi_oversold,
            min_confidence=self.settings.min_confidence,
        )

    def run_once(
        self,
        symbols: Optional[List[str]] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        selected_symbols = [item.upper() for item in (symbols or self.settings.symbols_list)]
        clock = self.client.get_clock()

        # Separate crypto and stock symbols
        crypto_symbols = [s for s in selected_symbols if is_crypto(s)]
        stock_symbols = [s for s in selected_symbols if not is_crypto(s)]

        # Only block stock trading when market is closed
        # Crypto trades 24/7 so it always runs
        market_open = clock.get("is_open", False)

        if not force and not market_open and not crypto_symbols:
            return {
                "executed": False,
                "reason": "market_closed",
                "clock": clock,
                "actions": [],
            }

        # If market closed, only run crypto symbols
        active_symbols = crypto_symbols if not market_open else selected_symbols
        if force:
            active_symbols = selected_symbols

        account = self.client.get_account()
        equity = float(account.get("equity", 0))
        cash = float(account.get("cash", 0))

        positions_list = self.client.list_positions()
        positions = {row["symbol"]: row for row in positions_list}
        actions: list[dict[str, Any]] = []

        for symbol in active_symbols:
            crypto = is_crypto(symbol)

            try:
                signal = self.get_signal(symbol)
            except Exception as exc:  # noqa: BLE001
                actions.append(
                    {
                        "symbol": symbol,
                        "decision": "ERROR",
                        "reason": str(exc),
                    }
                )
                continue

            existing = positions.get(symbol)

            if signal.action == "BUY":
                if existing:
                    actions.append(
                        {
                            "symbol": symbol,
                            "decision": "SKIP",
                            "reason": "position_exists",
                            "signal": signal.to_dict(),
                        }
                    )
                    continue

                if len(positions) >= self.settings.max_positions:
                    actions.append(
                        {
                            "symbol": symbol,
                            "decision": "SKIP",
                            "reason": "max_positions_reached",
                            "signal": signal.to_dict(),
                        }
                    )
                    continue

                if self._in_cooldown(symbol):
                    actions.append(
                        {
                            "symbol": symbol,
                            "decision": "SKIP",
                            "reason": "cooldown_active",
                            "signal": signal.to_dict(),
                        }
                    )
                    continue

                asset = self.client.get_asset(symbol)
                if not asset or not asset.get("tradable"):
                    actions.append(
                        {
                            "symbol": symbol,
                            "decision": "SKIP",
                            "reason": "asset_not_tradable",
                            "signal": signal.to_dict(),
                        }
                    )
                    continue

                buy_qty = self._compute_buy_qty(equity, cash, signal.price, crypto=crypto)
                if buy_qty <= 0:
                    actions.append(
                        {
                            "symbol": symbol,
                            "decision": "SKIP",
                            "reason": "insufficient_budget",
                            "signal": signal.to_dict(),
                        }
                    )
                    continue

                order: dict[str, Any] = {
                    "symbol": symbol,
                    "qty": self._format_qty(buy_qty, crypto=crypto),
                    "side": "buy",
                    "type": "market",
                    # Crypto uses "gtc" (good till cancelled), stocks use "day"
                    "time_in_force": "gtc" if crypto else self.settings.order_time_in_force,
                }

                # Bracket orders not supported for crypto — use plain market order
                if not crypto and self.settings.stop_loss_pct > 0 and self.settings.take_profit_pct > 0:
                    order["order_class"] = "bracket"
                    order["take_profit"] = {
                        "limit_price": round(
                            signal.price * (1 + self.settings.take_profit_pct), 2
                        )
                    }
                    order["stop_loss"] = {
                        "stop_price": round(
                            signal.price * (1 - self.settings.stop_loss_pct), 2
                        )
                    }

                if self.settings.dry_run:
                    response = {"dry_run": True, "order": order}
                else:
                    try:
                        response = self.client.submit_order(order)
                    except AlpacaAPIError as exc:
                        # If bracket pricing is rejected, fall back to plain market entry
                        if order.get("order_class") and (
                            "stop_loss.stop_price" in str(exc)
                            or "take_profit.limit_price" in str(exc)
                        ):
                            fallback_order = {
                                key: value
                                for key, value in order.items()
                                if key not in {"order_class", "take_profit", "stop_loss"}
                            }
                            response = self.client.submit_order(fallback_order)
                            response["_fallback"] = "submitted_without_bracket"
                        else:
                            raise

                actions.append(
                    {
                        "symbol": symbol,
                        "decision": "BUY",
                        "signal": signal.to_dict(),
                        "order": response,
                    }
                )
                self._mark_traded(symbol)
                positions[symbol] = {"symbol": symbol, "qty": order["qty"]}
                cash -= buy_qty * signal.price
                continue

            if signal.action == "SELL":
                if not existing:
                    actions.append(
                        {
                            "symbol": symbol,
                            "decision": "SKIP",
                            "reason": "no_position",
                            "signal": signal.to_dict(),
                        }
                    )
                    continue

                if existing.get("side") == "short":
                    actions.append(
                        {
                            "symbol": symbol,
                            "decision": "SKIP",
                            "reason": "short_positions_not_supported",
                            "signal": signal.to_dict(),
                        }
                    )
                    continue

                if self._in_cooldown(symbol):
                    actions.append(
                        {
                            "symbol": symbol,
                            "decision": "SKIP",
                            "reason": "cooldown_active",
                            "signal": signal.to_dict(),
                        }
                    )
                    continue

                sell_qty = self._position_qty(existing)
                if sell_qty <= 0:
                    actions.append(
                        {
                            "symbol": symbol,
                            "decision": "SKIP",
                            "reason": "invalid_position_qty",
                            "signal": signal.to_dict(),
                        }
                    )
                    continue

                order = {
                    "symbol": symbol,
                    "qty": self._format_qty(sell_qty, crypto=crypto),
                    "side": "sell",
                    "type": "market",
                    "time_in_force": "gtc" if crypto else self.settings.order_time_in_force,
                }
                if self.settings.dry_run:
                    response = {"dry_run": True, "order": order}
                else:
                    response = self.client.submit_order(order)

                actions.append(
                    {
                        "symbol": symbol,
                        "decision": "SELL",
                        "signal": signal.to_dict(),
                        "order": response,
                    }
                )
                self._mark_traded(symbol)
                positions.pop(symbol, None)
                continue

            actions.append(
                {
                    "symbol": symbol,
                    "decision": "HOLD",
                    "signal": signal.to_dict(),
                }
            )

        return {
            "executed": True,
            "clock": clock,
            "dry_run": self.settings.dry_run,
            "actions": actions,
        }
