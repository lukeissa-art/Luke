from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    alpaca_api_key: str = Field(..., alias="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(..., alias="ALPACA_API_SECRET")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        alias="ALPACA_BASE_URL",
    )
    alpaca_data_url: str = Field(
        default="https://data.alpaca.markets",
        alias="ALPACA_DATA_URL",
    )

    symbols: str = Field(default="SOL/USD", alias="SYMBOLS")
    bar_timeframe: str = Field(default="5Min", alias="BAR_TIMEFRAME")
    lookback_bars: int = Field(default=300, alias="LOOKBACK_BARS")
    fast_ema: int = Field(default=8, alias="FAST_EMA")
    slow_ema: int = Field(default=13, alias="SLOW_EMA")
    rsi_period: int = Field(default=10, alias="RSI_PERIOD")
    rsi_overbought: int = Field(default=75, alias="RSI_OVERBOUGHT")
    rsi_oversold: int = Field(default=30, alias="RSI_OVERSOLD")
    min_confidence: float = Field(default=0.35, alias="MIN_CONFIDENCE")
    volume_lookback: int = Field(default=30, alias="VOLUME_LOOKBACK")
    volume_spike_multiplier: float = Field(default=2.5, alias="VOLUME_SPIKE_MULTIPLIER")
    gap_threshold: float = Field(default=0.002, alias="GAP_THRESHOLD")
    min_body_pct: float = Field(default=0.003, alias="MIN_BODY_PCT")

    max_notional_per_position: float = Field(
        default=0.20,
        alias="MAX_NOTIONAL_PER_POSITION",
    )
    max_positions: int = Field(default=3, alias="MAX_POSITIONS")
    stop_loss_pct: float = Field(default=0.0075, alias="STOP_LOSS_PCT")
    take_profit_pct: float = Field(default=0.012, alias="TAKE_PROFIT_PCT")
    cooldown_minutes: int = Field(default=5, alias="COOLDOWN_MINUTES")

    dry_run: bool = Field(default=False, alias="DRY_RUN")
    order_time_in_force: str = Field(default="day", alias="ORDER_TIME_IN_FORCE")
    poll_interval_seconds: int = Field(default=60, alias="POLL_INTERVAL_SECONDS")

    @property
    def symbols_list(self) -> list[str]:
        return [item.strip().upper() for item in self.symbols.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
