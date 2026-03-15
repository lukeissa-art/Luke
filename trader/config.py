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

    symbols: str = Field(default="AAPL,MSFT,NVDA,SPY,QQQ", alias="SYMBOLS")
    bar_timeframe: str = Field(default="5Min", alias="BAR_TIMEFRAME")
    lookback_bars: int = Field(default=120, alias="LOOKBACK_BARS")
    fast_ema: int = Field(default=12, alias="FAST_EMA")
    slow_ema: int = Field(default=26, alias="SLOW_EMA")
    rsi_period: int = Field(default=14, alias="RSI_PERIOD")

    max_notional_per_position: float = Field(
        default=0.10,
        alias="MAX_NOTIONAL_PER_POSITION",
    )
    max_positions: int = Field(default=3, alias="MAX_POSITIONS")
    stop_loss_pct: float = Field(default=0.02, alias="STOP_LOSS_PCT")
    take_profit_pct: float = Field(default=0.04, alias="TAKE_PROFIT_PCT")
    cooldown_minutes: int = Field(default=30, alias="COOLDOWN_MINUTES")

    dry_run: bool = Field(default=True, alias="DRY_RUN")
    order_time_in_force: str = Field(default="day", alias="ORDER_TIME_IN_FORCE")
    poll_interval_seconds: int = Field(default=300, alias="POLL_INTERVAL_SECONDS")

    @property
    def symbols_list(self) -> list[str]:
        return [item.strip().upper() for item in self.symbols.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

