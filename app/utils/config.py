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
class LLMConfig:
    """Defines the LLM configuration settings for the application."""

    model: str = "qwen/qwen3.6-27b"
    temperature: float = 0.7
    max_tokens: int = 1000


@dataclass
class StmConfig:
    """Short-term memory thresholds that control the rolling-summary behaviour.

    Attributes:
        threshold: Total message count that triggers a summarisation pass.
        keep_recent: Number of recent messages kept in the live context window
            after trimming.  Older messages are folded into the rolling summary.
        summarize_step: How many additional messages must accumulate before the
            next summarisation pass is triggered after the current one.
    """

    threshold: int = 10
    keep_recent: int = 5
    summarize_step: int = 10


@dataclass
class AppConfig:
    """FastAPI configuration."""

    fastapi_url: str = "http://localhost:8000/api/v1/chat"
    fastapi_stream_url: str = "http://localhost:8000/api/v1/chat/stream"


secrets = Secrets()
llm_config = LLMConfig()
stm_config = StmConfig()
other = OtherConfig()
app_config = AppConfig()
