"""Single source of configuration. No ``os.getenv`` anywhere else in the tree."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CALLYTICS_",
        env_file=(".env", "/etc/callytics/callytics.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # identity
    env: str = "dev"
    timezone: str = "Asia/Kolkata"

    # storage
    mysql_dsn: str = "sqlite:///callytics-dev.sqlite3"
    mysql_pool_size: int = 5

    # event bus
    broker: str = "memory"  # memory|kafka
    kafka_bootstrap: str = "127.0.0.1:9092"
    topic_prefix: str = "callytics"

    # api
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    api_token: str = ""  # shared secret for service-to-service callers
    cors_origins: str = ""

    # FinEcho (conversation intelligence) — service boundary, not an import
    finecho_url: str = "http://127.0.0.1:8090"
    finecho_token: str = ""
    finecho_webhook_secret: str = ""
    finecho_timeout_s: int = 30

    # LeadSquared — read-only, used for migration and dual-run reconciliation
    lsq_base_url: str = "https://api-in21.leadsquared.com/v2"
    lsq_access_key: str = ""
    lsq_secret_key: str = ""
    lsq_rate_per_min: int = 50

    # ad-audience sync
    meta_ad_account_id: str = ""
    meta_access_token: str = ""
    google_customer_id: str = ""
    google_developer_token: str = ""
    audience_sync_enabled: bool = False

    # intelligence tuning
    stage_proposal_auto_apply: bool = False
    stage_proposal_min_confidence: float = 0.80
    score_half_life_days: float = 14.0
    working_hours_start: int = 9
    working_hours_end: int = 20

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
