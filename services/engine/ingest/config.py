"""Ingest configuration loaded from environment variables."""

import os


class IngestConfig:
    """Configuration for ingestion jobs, loaded from environment."""

    def __init__(self) -> None:
        self.database_url: str = os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg://predictor:changeme@db:5432/football_predictor",
        )
        self.redis_url: str = os.environ.get("REDIS_URL", "redis://redis:6379/0")
        self.football_data_org_token: str = os.environ.get("FOOTBALL_DATA_ORG_TOKEN", "")
        self.football_data_org_base_url: str = os.environ.get(
            "FOOTBALL_DATA_ORG_BASE_URL", "https://api.football-data.org/v4"
        )
        self.football_data_couk_base_url: str = os.environ.get(
            "FOOTBALL_DATA_COUK_BASE_URL",
            "https://www.football-data.co.uk/mmz4281",
        )
        self.raw_data_dir: str = os.environ.get("RAW_DATA_DIR", "/app/data/raw")
        self.fuzzy_auto_accept_threshold: float = float(
            os.environ.get("FUZZY_AUTO_ACCEPT_THRESHOLD", "92")
        )
        self.fuzzy_min_gap: float = float(os.environ.get("FUZZY_MIN_GAP", "3"))
        self.fd_org_rate_limit: int = int(os.environ.get("FD_ORG_RATE_LIMIT", "10"))
        self.fd_org_daily_budget: int = int(os.environ.get("FD_ORG_DAILY_BUDGET", "500"))
