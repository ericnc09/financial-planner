from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # API Keys (edgar + congress are free, no keys needed)
    tiingo_api_key: str = ""
    fred_api_key: str = ""

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

    log_level: str = "INFO"

    model_config = {"env_file": "config/.env", "env_file_encoding": "utf-8"}
