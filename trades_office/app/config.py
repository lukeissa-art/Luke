import os
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    company_name: str = "TradeDesk AI"
    public_base_url: str = "http://localhost:8000"
    database_path: str = "trades_office.db"
    run_worker: bool = True  # run follow-up/report jobs inside the web process
    seed_demo: bool = False  # create the demo shop on startup if it's missing

    # Dashboard login (HTTP basic auth). Change before deploying.
    admin_username: str = "admin"
    admin_password: str = "change-me"

    # Claude
    anthropic_model: str = "claude-sonnet-5"
    # Phone calls are latency-sensitive: "low" keeps replies fast. Raise if quality suffers.
    anthropic_effort: str = "low"
    # Server-side refusal fallback (beta). Off by default: turn on only for a model that supports it.
    anthropic_fallbacks: bool = False

    # Twilio (leave blank to run in console/dry-run mode)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_validate_signatures: bool = True
    tts_voice: str = "Polly.Joanna-Neural"

    # Stripe
    stripe_api_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_starter: str = ""
    stripe_price_pro: str = ""
    stripe_price_plus: str = ""

    # Google Calendar (service account JSON, shared with each shop's calendar)
    google_service_account_file: str = ""

    # Public website
    contact_email: str = "hello@example.com"
    founder_phone: str = ""  # gets a text for every new trial signup
    company_sms_number: str = ""  # Twilio number the founder alerts are sent from
    public_demo_shop_id: int = 0  # shop the website's "try it" demo talks to; 0 turns the demo off
    demo_phone_number: str = ""  # optional real demo line shown on the website
    service_region: str = "Central Texas"

    # Follow-up timing
    unbooked_followup_minutes: int = 15
    quote_followup_days: int = 2
    review_request_hours: int = 2

    @model_validator(mode="after")
    def _detect_host_url(self):
        # Pick up the public URL Render / Railway assign, unless PUBLIC_BASE_URL is set explicitly.
        if "PUBLIC_BASE_URL" not in os.environ:
            if os.environ.get("RENDER_EXTERNAL_URL"):
                self.public_base_url = os.environ["RENDER_EXTERNAL_URL"]
            elif os.environ.get("RAILWAY_PUBLIC_DOMAIN"):
                self.public_base_url = "https://" + os.environ["RAILWAY_PUBLIC_DOMAIN"]
        self.public_base_url = self.public_base_url.rstrip("/")
        return self

    @property
    def is_local(self) -> bool:
        return self.public_base_url.startswith(("http://localhost", "http://127.0.0.1"))

    @property
    def twilio_enabled(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token)


@lru_cache
def get_settings() -> Settings:
    return Settings()
