"""
Backtest script for the Alpaca trading bot strategy.
Uses historical data from Alpaca to simulate trades.

Usage:
    python3 backtest.py

Make sure your .env file is set up with your Alpaca API keys.
"""

from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timedelta, timezone
from itertools import product
from typing import Any
import time

import requests
from dotenv import load_dotenv
from trader.strategy import generate_signal

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY        = os.getenv("ALPACA_API_KEY", "")
API_SECRET     = os.getenv("ALPACA_API_SECRET", "")
DATA_URL       = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

# Symbols can be overridden via env SYMBOLS="..." or BACKTEST_SYMBOLS; default SOL only
SYMBOLS        = [s.strip() for s in os.getenv("BACKTEST_SYMBOLS", os.getenv("SYMBOLS", "SOL/USD")).split(",") if s.strip()]
TIMEFRAME      = os.getenv("BAR_TIMEFRAME", "5Min")
LOOKBACK_DAYS  = int(os.getenv("LOOKBACK_DAYS", "240"))          # used when RANDOM_WINDOW is false
WINDOW_DAYS    = int(os.getenv("WINDOW_DAYS", "90"))             # random window length
WINDOW_YEARS   = int(os.getenv("WINDOW_YEARS", "2"))             # pick start within last N years
RANDOM_WINDOW  = os.getenv("RANDOM_WINDOW", "true").lower() in {"1","true","yes","on"}
# Optimizer can use separate window settings; defaults mirror backtest
OPT_WINDOW_DAYS  = int(os.getenv("OPT_WINDOW_DAYS", str(WINDOW_DAYS)))
OPT_WINDOW_YEARS = int(os.getenv("OPT_WINDOW_YEARS", str(WINDOW_YEARS)))
OPT_RANDOM_WINDOW = os.getenv("OPT_RANDOM_WINDOW", str(RANDOM_WINDOW)).lower() in {"1","true","yes","on"}
FAST_EMA       = 8
SLOW_EMA       = 13
RSI_PERIOD     = 10
RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT", "75"))
RSI_OVERSOLD   = int(os.getenv("RSI_OVERSOLD", "30"))
# Looser defaults in backtest to ensure trades occur; can be tightened via env.
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.15"))
VOLUME_LOOKBACK = int(os.getenv("VOLUME_LOOKBACK", "20"))
VOLUME_SPIKE_MULTIPLIER = float(os.getenv("VOLUME_SPIKE_MULTIPLIER", "1.5"))
GAP_THRESHOLD = float(os.getenv("GAP_THRESHOLD", "0.001"))
MIN_BODY_PCT = float(os.getenv("MIN_BODY_PCT", "0.0015"))
STOP_LOSS_PCT  = 0.0075
TAKE_PROFIT_PCT = 0.012
STARTING_CASH  = 100_000.0
POSITION_SIZE  = 0.20                      # 20% of equity per trade
MAX_POSITIONS  = 3
COOLDOWN_BARS  = 3                         # bars to wait after a trade

# Targets for optimization
TARGET_RETURN_PCT = 0.5
TARGET_WIN_RATE   = 55.0

# Network / retry tuning
REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
BACKOFF_SECONDS = [2, 4, 8, 15]

# Optimizer limits
MAX_COMBOS = int(os.getenv("OPT_MAX_COMBOS", "60"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_crypto_bars(
    symbol: str,
    days: int,
    timeframe: str,
    *,
    random_window: bool | None = None,
    window_days: int | None = None,
    window_years: int | None = None,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    use_random = RANDOM_WINDOW if random_window is None else random_window
    win_days = WINDOW_DAYS if window_days is None else window_days
    win_years = WINDOW_YEARS if window_years is None else window_years

    if use_random:
        max_start = now - timedelta(days=win_days)
        min_start = now - timedelta(days=win_years * 365)
        if max_start <= min_start:
            start = min_start
        else:
            span_seconds = (max_start - min_start).total_seconds()
            start = min_start + timedelta(seconds=random.random() * span_seconds)
        end = start + timedelta(days=win_days)
        if end > now:
            end = now
            start = end - timedelta(days=win_days)
        print(f"[window] {symbol} {start.isoformat()} -> {end.isoformat()} (random)")
    else:
        end = now
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
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 1000,
        }
        if next_page_token:
            params["page_token"] = next_page_token

        response = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                break
            except requests.exceptions.RequestException as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                sleep_for = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                print(f"  Retry {attempt+1}/{MAX_RETRIES} for {symbol} after error {exc}; sleeping {sleep_for}s")
                time.sleep(sleep_for)

        if not response or not response.ok:
            status = response.status_code if response else "no_response"
            text = response.text if response else ""
            print(f"  Error fetching {symbol}: {status} {text}")
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


# ── Backtest engine ───────────────────────────────────────────────────────────

def backtest_symbol(
    symbol: str,
    bars: list[dict],
    *,
    fast_ema: int,
    slow_ema: int,
    rsi_period: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    position_size: float,
    cooldown_bars: int,
) -> dict:
    closes = [float(b["c"]) for b in bars]
    opens  = [float(b["o"]) for b in bars]
    volumes = [float(b.get("v", 0)) for b in bars]
    times  = [b["t"] for b in bars]

    cash       = STARTING_CASH
    equity     = STARTING_CASH
    position   = 0.0       # qty held
    entry_price = 0.0
    trades     = []
    cooldown   = 0
    peak_equity = STARTING_CASH
    max_drawdown = 0.0

    min_bars = max(slow_ema + 2, rsi_period + 2)

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
            if pnl_pct <= -stop_loss_pct:
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
                cooldown = cooldown_bars
                continue
            elif pnl_pct >= take_profit_pct:
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
                cooldown = cooldown_bars
                continue

        if cooldown > 0:
            cooldown -= 1
            continue

        opens_window = opens[:i+1]
        vols_window = volumes[:i+1]
        try:
            signal_obj = generate_signal(
                symbol=symbol,
                closes=window,
                opens=opens_window,
                volumes=vols_window,
                fast_period=fast_ema,
                slow_period=slow_ema,
                rsi_period=rsi_period,
                rsi_overbought=RSI_OVERBOUGHT,
                rsi_oversold=RSI_OVERSOLD,
                min_confidence=MIN_CONFIDENCE,
                volume_lookback=VOLUME_LOOKBACK,
                volume_spike_multiplier=VOLUME_SPIKE_MULTIPLIER,
                gap_threshold=GAP_THRESHOLD,
                min_body_pct=MIN_BODY_PCT,
            )
            signal = signal_obj.action
        except ValueError:
            signal = "HOLD"

        if signal == "BUY" and position == 0:
            budget = min(equity * position_size, cash * 0.95)
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
            cooldown = cooldown_bars

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


def run_params_backtest(
    params: dict[str, Any],
    *,
    random_window: bool | None = None,
    window_days: int | None = None,
    window_years: int | None = None,
    bars_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """
    Execute backtest across all symbols for a parameter set.
    """
    all_results: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        try:
            if bars_cache and symbol in bars_cache:
                bars = bars_cache[symbol]
            else:
                bars = fetch_crypto_bars(
                    symbol,
                    LOOKBACK_DAYS,
                    params["timeframe"],
                    random_window=random_window,
                    window_days=window_days,
                    window_years=window_years,
                )
        except Exception as exc:  # noqa: BLE001 - skip symbol on fetch failure
            print(f"  Skipping {symbol} due to fetch error: {exc}")
            continue
        if len(bars) < 50:
            continue
        result = backtest_symbol(
            symbol,
            bars,
            fast_ema=params["fast_ema"],
            slow_ema=params["slow_ema"],
            rsi_period=params["rsi_period"],
            stop_loss_pct=params["stop_loss_pct"],
            take_profit_pct=params["take_profit_pct"],
            position_size=params["position_size"],
            cooldown_bars=params["cooldown_bars"],
        )
        all_results.append(result)

    avg_return = (
        sum(r["total_return_pct"] for r in all_results) / len(all_results)
        if all_results else 0
    )
    avg_win_rate = (
        sum(r["win_rate"] for r in all_results) / len(all_results)
        if all_results else 0
    )

    return {
        "params": params,
        "results": all_results,
        "avg_return_pct": round(avg_return, 2),
        "avg_win_rate": round(avg_win_rate, 2),
        "total_trades": sum(r["total_trades"] for r in all_results),
    }


def optimize(
    *,
    random_window: bool | None = None,
    window_days: int | None = None,
    window_years: int | None = None,
) -> dict[str, Any]:
    """
    Grid search for parameters that hit target return and win rate.
    """
    param_grid = {
        "timeframe": ["5Min"],
        "fast_ema": [5, 8],
        "slow_ema": [13, 21],
        "rsi_period": [7, 10],
        "stop_loss_pct": [0.0075, 0.01],
        "take_profit_pct": [0.012, 0.018],
        "position_size": [0.10, 0.20],
        "cooldown_bars": [3, 5],
    }

    # Fetch bars once per symbol to speed up search and ensure a consistent window
    bars_cache: dict[str, list[dict[str, Any]]] = {}
    for symbol in SYMBOLS:
        try:
            bars_cache[symbol] = fetch_crypto_bars(
                symbol,
                LOOKBACK_DAYS,
                "5Min",
                random_window=random_window,
                window_days=window_days,
                window_years=window_years,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Skipping {symbol} in optimizer due to fetch error: {exc}")

    best: dict[str, Any] | None = None
    combos = list(product(*param_grid.values()))
    total = min(len(combos), MAX_COMBOS)
    for idx, combo in enumerate(combos[:total], 1):
        params = dict(zip(param_grid.keys(), combo))
        summary = run_params_backtest(
            params,
            random_window=random_window,
            window_days=window_days,
            window_years=window_years,
            bars_cache=bars_cache,
        )
        if (
            summary["avg_return_pct"] >= TARGET_RETURN_PCT
            and summary["avg_win_rate"] >= TARGET_WIN_RATE
        ):
            if best is None or summary["avg_return_pct"] > best["avg_return_pct"]:
                best = summary
        # Early break if we already found a strong candidate with high return
        if best and best["avg_return_pct"] > TARGET_RETURN_PCT + 1.0:
            break
        print(
            f"[{idx}/{total}] "
            f"return {summary['avg_return_pct']}% "
            f"win {summary['avg_win_rate']}% "
            f"params {params}"
        )

    return best or {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    optimize_flag = "--optimize" in sys.argv or os.getenv("OPTIMIZE", "0") == "1"

    print("=" * 60)
    print("  BACKTEST RESULTS")
    if RANDOM_WINDOW:
        print(f"  Period: Random {WINDOW_DAYS}-day window within last {WINDOW_YEARS} years")
    else:
        print(f"  Period: Last {LOOKBACK_DAYS} days")
    print("=" * 60)

    if optimize_flag:
        print(f"Running parameter search for targets "
              f"return>={TARGET_RETURN_PCT}% win_rate>={TARGET_WIN_RATE}% ...")
        best = optimize(
            random_window=OPT_RANDOM_WINDOW,
            window_days=OPT_WINDOW_DAYS,
            window_years=OPT_WINDOW_YEARS,
        )
        if not best:
            print("No parameter set hit the targets. Try widening the grid or timeframe.")
            return
        params = best["params"]
        print("\nBest parameters found:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print(f"Average return: {best['avg_return_pct']}%")
        print(f"Average win rate: {best['avg_win_rate']}%")
        return

    params = {
        "timeframe": TIMEFRAME,
        "fast_ema": FAST_EMA,
        "slow_ema": SLOW_EMA,
        "rsi_period": RSI_PERIOD,
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "position_size": POSITION_SIZE,
        "cooldown_bars": COOLDOWN_BARS,
    }

    summary = run_params_backtest(params)
    all_results = summary["results"]

    for result in all_results:
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

        if result["trades"]:
            print(f"\n  Last 5 trades:")
            for t in result["trades"][-5:]:
                sign = "+" if t["pnl_pct"] > 0 else ""
                print(f"    {t['reason']:15} entry ${t['entry']:,.2f}  exit ${t['exit']:,.2f}  {sign}{t['pnl_pct']}%")

    if all_results:
        print(f"\n{'=' * 60}")
        print("  OVERALL SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total trades across all symbols: {summary['total_trades']}")
        print(f"  Average win rate:                {summary['avg_win_rate']}%")
        print(f"  Average return:                  {summary['avg_return_pct']}%")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
