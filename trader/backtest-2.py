"""
Backtest script for the Alpaca trading bot strategy.
Uses historical data from Alpaca to simulate trades.

Usage:
    python3 backtest.py

Make sure your .env file is set up with your Alpaca API keys.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY        = os.getenv("ALPACA_API_KEY", "")
API_SECRET     = os.getenv("ALPACA_API_SECRET", "")
DATA_URL       = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

SYMBOLS        = ["BTC/USD", "ETH/USD", "SOL/USD"]   # symbols to backtest
TIMEFRAME      = "1Hour"
LOOKBACK_DAYS  = 90                        # how far back to test (days)
FAST_EMA       = 12
SLOW_EMA       = 26
RSI_PERIOD     = 14
STOP_LOSS_PCT  = 0.015
TAKE_PROFIT_PCT = 0.05
STARTING_CASH  = 100_000.0
POSITION_SIZE  = 0.02                      # 2% of equity per trade
MAX_POSITIONS  = 3
COOLDOWN_BARS  = 15                        # bars to wait after a trade

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_crypto_bars(symbol: str, days: int) -> list[dict[str, Any]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }
    url = f"{DATA_URL}/v1beta3/crypto/us/bars"
    all_bars = []
    next_page_token = None

    while True:
        params = {
            "symbols": symbol,
            "timeframe": TIMEFRAME,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
        }
        if next_page_token:
            params["page_token"] = next_page_token

        response = requests.get(url, headers=headers, params=params, timeout=30)
        if not response.ok:
            print(f"  Error fetching {symbol}: {response.status_code} {response.text}")
            break

        data = response.json()
        bars = data.get("bars", {}).get(symbol, [])
        all_bars.extend(bars)

        next_page_token = data.get("next_page_token")
        if not next_page_token:
            break

        print(f"  Fetched {len(all_bars)} bars so far for {symbol}...")

    print(f"  Fetched {len(all_bars)} total bars for {symbol}")
    return all_bars


def ema_series(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    multiplier = 2 / (period + 1)
    result = []
    current = sum(values[:period]) / period
    result.append(current)
    for value in values[period:]:
        current = (value - current) * multiplier + current
        result.append(current)
    return result


def rsi_value(values: list[float], period: int) -> float:
    if len(values) <= period:
        return 50.0
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def bollinger_bands(values: list[float], period: int, num_std: float = 2.0):
    if len(values) < period:
        return None, None, None
    window = values[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = variance ** 0.5
    return middle + num_std * std, middle, middle - num_std * std


def detect_market_regime(closes: list[float], period: int = 50) -> str:
    if len(closes) < period:
        return "ranging"
    window = closes[-period:]
    high = max(window)
    low  = min(window)
    price_range = (high - low) / low if low > 0 else 0
    return "trending" if price_range > 0.03 else "ranging"


def get_signal(closes: list[float]) -> str:
    bb_period = 20
    min_bars  = max(SLOW_EMA + 2, RSI_PERIOD + 2, bb_period + 2)
    if len(closes) < min_bars:
        return "HOLD"

    fast_s = ema_series(closes, FAST_EMA)
    slow_s = ema_series(closes, SLOW_EMA)
    offset = len(fast_s) - len(slow_s)
    aligned = fast_s[offset:]

    fast_now, fast_prev = aligned[-1], aligned[-2]
    slow_now, slow_prev = slow_s[-1], slow_s[-2]

    bullish_cross = fast_prev <= slow_prev and fast_now > slow_now
    bearish_cross  = fast_prev >= slow_prev and fast_now < slow_now

    rsi        = rsi_value(closes, RSI_PERIOD)
    last_price = closes[-1]
    uptrend    = fast_now > slow_now

    upper_bb, middle_bb, lower_bb = bollinger_bands(closes, bb_period)
    if upper_bb is None:
        return "HOLD"

    below_lower_bb = last_price < lower_bb
    above_upper_bb = last_price > upper_bb
    regime = detect_market_regime(closes)

    if regime == "ranging":
        if below_lower_bb and rsi <= 35 and rsi > 20:
            return "BUY"
        elif above_upper_bb and rsi >= 65:
            return "SELL"
        elif last_price > middle_bb and rsi >= 70:
            return "SELL"

    elif regime == "trending":
        if bullish_cross and last_price > middle_bb and rsi < 70 and rsi > 30:
            return "BUY"
        elif uptrend and below_lower_bb and rsi <= 40:
            return "BUY"
        elif bearish_cross and rsi < 50:
            return "SELL"
        elif above_upper_bb and rsi >= 75:
            return "SELL"

    return "HOLD"


# ── Backtest engine ───────────────────────────────────────────────────────────

def backtest_symbol(symbol: str, bars: list[dict]) -> dict:
    closes = [float(b["c"]) for b in bars]
    times  = [b["t"] for b in bars]

    cash       = STARTING_CASH
    equity     = STARTING_CASH
    position   = 0.0       # qty held
    entry_price = 0.0
    trades     = []
    cooldown   = 0
    peak_equity = STARTING_CASH
    max_drawdown = 0.0

    min_bars = max(SLOW_EMA + 2, RSI_PERIOD + 2)

    for i in range(min_bars, len(closes)):
        price  = closes[i]
        window = closes[:i+1]

        # Update equity mark-to-market
        equity = cash + position * price
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

        # Check stop loss / take profit on open position
        if position > 0:
            pnl_pct = (price - entry_price) / entry_price
            if pnl_pct <= -STOP_LOSS_PCT:
                proceeds = position * price
                cash += proceeds
                trades.append({
                    "type": "SELL",
                    "reason": "stop_loss",
                    "entry": entry_price,
                    "exit": price,
                    "pnl_pct": round(pnl_pct * 100, 3),
                    "time": times[i],
                })
                position = 0.0
                entry_price = 0.0
                cooldown = COOLDOWN_BARS
                continue
            elif pnl_pct >= TAKE_PROFIT_PCT:
                proceeds = position * price
                cash += proceeds
                trades.append({
                    "type": "SELL",
                    "reason": "take_profit",
                    "entry": entry_price,
                    "exit": price,
                    "pnl_pct": round(pnl_pct * 100, 3),
                    "time": times[i],
                })
                position = 0.0
                entry_price = 0.0
                cooldown = COOLDOWN_BARS
                continue

        if cooldown > 0:
            cooldown -= 1
            continue

        signal = get_signal(window)

        if signal == "BUY" and position == 0:
            budget = min(equity * POSITION_SIZE, cash * 0.95)
            qty    = budget / price
            if qty > 0:
                cost = qty * price
                cash -= cost
                position = qty
                entry_price = price

        elif signal == "SELL" and position > 0:
            proceeds = position * price
            pnl_pct  = (price - entry_price) / entry_price
            cash += proceeds
            trades.append({
                "type": "SELL",
                "reason": "signal",
                "entry": entry_price,
                "exit": price,
                "pnl_pct": round(pnl_pct * 100, 3),
                "time": times[i],
            })
            position = 0.0
            entry_price = 0.0
            cooldown = COOLDOWN_BARS

    # Close any open position at end
    if position > 0:
        final_price = closes[-1]
        pnl_pct = (final_price - entry_price) / entry_price
        cash += position * final_price
        trades.append({
            "type": "SELL",
            "reason": "end_of_backtest",
            "entry": entry_price,
            "exit": final_price,
            "pnl_pct": round(pnl_pct * 100, 3),
            "time": times[-1],
        })

    final_equity = cash
    total_return = (final_equity - STARTING_CASH) / STARTING_CASH * 100

    winning = [t for t in trades if t["pnl_pct"] > 0]
    losing  = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(winning) / len(trades) * 100 if trades else 0
    avg_win  = sum(t["pnl_pct"] for t in winning) / len(winning) if winning else 0
    avg_loss = sum(t["pnl_pct"] for t in losing)  / len(losing)  if losing  else 0

    return {
        "symbol": symbol,
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "total_return_pct": round(total_return, 2),
        "final_equity": round(final_equity, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "trades": trades,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  BACKTEST RESULTS")
    print(f"  Period: Last {LOOKBACK_DAYS} days  |  Timeframe: {TIMEFRAME}")
    print(f"  Strategy: EMA {FAST_EMA}/{SLOW_EMA} + RSI {RSI_PERIOD}")
    print(f"  Stop Loss: {STOP_LOSS_PCT*100}%  |  Take Profit: {TAKE_PROFIT_PCT*100}%")
    print("=" * 60)

    all_results = []

    for symbol in SYMBOLS:
        print(f"\nFetching data for {symbol}...")
        bars = fetch_crypto_bars(symbol, LOOKBACK_DAYS)
        if len(bars) < 50:
            print(f"  Not enough data for {symbol}, skipping.")
            continue

        result = backtest_symbol(symbol, bars)
        all_results.append(result)

        print(f"\n{'─' * 40}")
        print(f"  {result['symbol']}")
        print(f"{'─' * 40}")
        print(f"  Total trades:    {result['total_trades']}")
        print(f"  Win rate:        {result['win_rate']}%")
        print(f"  Winning trades:  {result['winning_trades']}")
        print(f"  Losing trades:   {result['losing_trades']}")
        print(f"  Avg win:         +{result['avg_win_pct']}%")
        print(f"  Avg loss:        {result['avg_loss_pct']}%")
        print(f"  Max drawdown:    {result['max_drawdown_pct']}%")
        print(f"  Total return:    {result['total_return_pct']}%")
        print(f"  Final equity:    ${result['final_equity']:,.2f}")

        # Show last 5 trades
        if result["trades"]:
            print(f"\n  Last 5 trades:")
            for t in result["trades"][-5:]:
                sign = "+" if t["pnl_pct"] > 0 else ""
                print(f"    {t['reason']:15} entry ${t['entry']:,.2f}  exit ${t['exit']:,.2f}  {sign}{t['pnl_pct']}%")

    if all_results:
        print(f"\n{'=' * 60}")
        print("  OVERALL SUMMARY")
        print(f"{'=' * 60}")
        total_trades = sum(r["total_trades"] for r in all_results)
        avg_return   = sum(r["total_return_pct"] for r in all_results) / len(all_results)
        avg_win_rate = sum(r["win_rate"] for r in all_results) / len(all_results)
        print(f"  Total trades across all symbols: {total_trades}")
        print(f"  Average win rate:                {round(avg_win_rate, 1)}%")
        print(f"  Average return:                  {round(avg_return, 2)}%")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
