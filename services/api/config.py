from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://predictor:changeme@db:5432/football_predictor"
    redis_url: str = "redis://redis:6379/0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_log_level: str = "info"
    environment: str = "development"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
