from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    app_name: str = "agentic-model-risk-validator"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    use_llm: bool = Field(default=True, alias="USE_LLM")
    storage_dir: Path = Field(default=APP_DIR / "storage", alias="STORAGE_DIR")
    database_path: Path = Field(
        default=APP_DIR / "storage" / "model_validator.sqlite",
        alias="DATABASE_PATH",
    )
    allowed_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

