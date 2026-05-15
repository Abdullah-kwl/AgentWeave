"""This module defines the configuration for the application,
including secrets and other settings.
we use pydantic for secrets that we want to load from env vars,
and dataclass for other config that we want to hardcode in the codebase."""

from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Defines the secrets required for the application, which are loaded from environment variables."""

    GROQ_API_KEY: str
    GEMINI_API_KEY: str

    # load vars from .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")


class OtherConfig(BaseModel):
    """Defines other configuration settings for the application."""

    Samplefoo: str = "Samplebar"


@dataclass
class Config:
    """Defines the main configuration settings for the application."""

    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1000


@dataclass
class AppConfig:
    """FastAPI configuration."""

    fastapi_url: str = "http://localhost:8000/api/v1/chat"
    fastapi_stream_url: str = "http://localhost:8000/api/v1/chat/stream"


secrets = Secrets()
config = Config()
other = OtherConfig()
app_config = AppConfig()
