from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass
class Signal:
    symbol: str
    action: str
    price: float
    confidence: float
    reason: str
    fast_ema: float
    slow_ema: float
    rsi: float

    def to_dict(self) -> dict:
        return asdict(self)


# ── Indicators ────────────────────────────────────────────────────────────────

def ema(values: list[float], period: int) -> float:
    if period <= 0:
        raise ValueError("EMA period must be positive")
    if len(values) < period:
        raise ValueError("Not enough values for EMA")
    multiplier = 2 / (period + 1)
    current = sum(values[:period]) / period
    for value in values[period:]:
        current = (value - current) * multiplier + current
    return current


def ema_series(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        raise ValueError("Not enough values for EMA series")
    multiplier = 2 / (period + 1)
    current = sum(values[:period]) / period
    result = [current]
    for value in values[period:]:
        current = (value - current) * multiplier + current
        result.append(current)
    return result


def rsi(values: list[float], period: int) -> float:
    if period <= 0:
        raise ValueError("RSI period must be positive")
    if len(values) <= period:
        raise ValueError("Not enough values for RSI")
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


def bollinger_bands(values: list[float], period: int, num_std: float = 2.0) -> Tuple[float, float, float]:
    """Returns (upper_band, middle_band, lower_band)."""
    if len(values) < period:
        raise ValueError("Not enough values for Bollinger Bands")
    window = values[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = variance ** 0.5
    return middle + num_std * std, middle, middle - num_std * std


def detect_market_regime(closes: list[float], period: int = 50) -> str:
    """
    Detect if market is trending or ranging.
    Returns 'trending' or 'ranging'.
    """
    if len(closes) < period:
        return "ranging"
    window = closes[-period:]
    high = max(window)
    low  = min(window)
    price_range = (high - low) / low if low > 0 else 0
    return "trending" if price_range > 0.03 else "ranging"


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signal(
    symbol: str,
    closes: list[float],
    opens: list[float],
    volumes: list[float],
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    rsi_overbought: int,
    rsi_oversold: int,
    min_confidence: float,
    volume_lookback: int,
    volume_spike_multiplier: float,
    gap_threshold: float,
    min_body_pct: float,
) -> Signal:
    bb_period  = 20
    minimum    = max(slow_period + 2, rsi_period + 2, bb_period + 2)

    if len(closes) < minimum:
        raise ValueError(
            f"Need at least {minimum} bars for {symbol}, received {len(closes)}"
        )

    last_price = closes[-1]
    prev_price = closes[-2]

    # Core indicators
    fast_s = ema_series(closes, fast_period)
    slow_s = ema_series(closes, slow_period)
    offset = len(fast_s) - len(slow_s)
    aligned_fast = fast_s[offset:]

    fast_now  = aligned_fast[-1]
    fast_prev = aligned_fast[-2]
    slow_now  = slow_s[-1]
    slow_prev = slow_s[-2]

    rsi_val = rsi(closes, rsi_period)
    upper_bb, middle_bb, lower_bb = bollinger_bands(closes, bb_period)

    # Market regime
    regime = detect_market_regime(closes)

    # Crossovers
    bullish_cross = fast_prev <= slow_prev and fast_now > slow_now
    bearish_cross  = fast_prev >= slow_prev and fast_now < slow_now

    # Bollinger Band positions
    below_lower_bb = last_price < lower_bb
    above_upper_bb = last_price > upper_bb

    # Confidence
    gap_pct    = (fast_now - slow_now) / slow_now if slow_now else 0.0
    confidence = min(1.0, abs(gap_pct) * 50 + abs(rsi_val - 50.0) / 60.0)
    confidence = round(confidence, 3)

    action = "HOLD"
    reason = "No clear signal — waiting for confirmation"
    if confidence < min_confidence:
        return Signal(
            symbol=symbol.upper(),
            action=action,
            price=round(last_price, 4),
            confidence=confidence,
            reason="Below confidence threshold",
            fast_ema=round(fast_now, 4),
            slow_ema=round(slow_now, 4),
            rsi=round(rsi_val, 2),
        )

    # Volume context (not a hard gate, just boosts confidence)
    vol_window = volumes[-volume_lookback:] if volumes else []
    avg_vol = sum(vol_window) / len(vol_window) if vol_window else 0
    last_vol = volumes[-1] if volumes else 0
    vol_ratio = (last_vol / avg_vol) if avg_vol > 0 else 1.0
    body_pct = abs(last_price - opens[-1]) / opens[-1] if opens else 0
    impulse_up = (last_price - prev_price) / prev_price >= gap_threshold and body_pct >= min_body_pct
    impulse_down = (prev_price - last_price) / prev_price >= gap_threshold and body_pct >= min_body_pct

    # ── Momentum / trend following ───────────────────────────────────────────
    uptrend = fast_now > slow_now
    downtrend = fast_now < slow_now

    if bullish_cross and rsi_val >= max(40, rsi_oversold + 5):
        action = "BUY"
        reason = "Bullish EMA cross with momentum"
    elif uptrend and last_price > slow_now and 50 <= rsi_val <= max(70, rsi_overbought - 5):
        action = "BUY"
        reason = "Uptrend continuation"
    elif impulse_up and uptrend and rsi_val > 55:
        action = "BUY"
        reason = "Impulse in trend"
    elif bearish_cross and rsi_val <= min(60, rsi_overbought):
        action = "SELL"
        reason = "Bearish EMA cross"
    elif downtrend and last_price < slow_now and rsi_val < 45:
        action = "SELL"
        reason = "Downtrend continuation"
    elif impulse_down and rsi_val > rsi_overbought - 5:
        action = "SELL"
        reason = "Impulse against longs"
    elif rsi_val >= rsi_overbought + 2:
        action = "SELL"
        reason = "RSI overbought exit"

    # Boost confidence modestly when volume is elevated
    if vol_ratio > 1:
        confidence = min(1.0, confidence + (min(vol_ratio, 3.0) - 1) * 0.1)

    return Signal(
        symbol=symbol.upper(),
        action=action,
        price=round(last_price, 4),
        confidence=confidence,
        reason=reason,
        fast_ema=round(fast_now, 4),
        slow_ema=round(slow_now, 4),
        rsi=round(rsi_val, 2),
    )
