from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # API Keys (edgar + congress are free, no keys needed)
    tiingo_api_key: str = ""
    fred_api_key: str = ""
    finnhub_api_key: str | None = None  # free tier: 60 req/min — news primary
    bls_api_key: str | None = None      # optional; raises BLS quota 25→500/day

    # Database
    database_url: str = "sqlite:///./smart_money.db"

    # Scoring weights
    conviction_threshold: float = 0.6
    signal_weight: float = 0.4
    fundamental_weight: float = 0.35
    macro_weight: float = 0.25

    # Pipeline
    pipeline_mode: Literal["oneshot", "daemon"] = "oneshot"
    schedule_interval_minutes: int = 60

    # Production hardening
    # Mutating API endpoints (pipeline run, perf update, model train) require
    # this token via X-Admin-Token header. Unset = those endpoints are disabled.
    admin_api_token: str | None = None
    sentry_dsn: str | None = None  # error monitoring; unset = Sentry disabled

    # Alerts
    slack_webhook_url: str | None = None

    # Slack Bot (Socket Mode)
    slack_bot_token: str | None = None
    slack_app_token: str | None = None

    log_level: str = "INFO"

    model_config = {"env_file": "config/.env", "env_file_encoding": "utf-8"}
