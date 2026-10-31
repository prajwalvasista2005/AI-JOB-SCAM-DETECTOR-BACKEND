
from typing import Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    # Timeout for Gemini API requests in milliseconds. Must be >= 1.
    # Default: 30000 ms (30 seconds).
    GEMINI_REQUEST_TIMEOUT_MS: int = 30000

    # Explicit list of trusted proxy IP addresses. If empty, X-Forwarded-For is ignored.
    TRUSTED_PROXIES: list[str] = []

    # URL Ingestion Settings
    URL_FETCH_TIMEOUT_SECONDS: float = 10.0
    URL_FETCH_CONNECT_TIMEOUT_SECONDS: float = 3.0
    URL_FETCH_READ_TIMEOUT_SECONDS: float = 5.0
    URL_FETCH_MAX_BYTES: int = 2 * 1024 * 1024  # 2 MiB
    URL_FETCH_MAX_REDIRECTS: int = 3
    URL_FETCH_MAX_TEXT_LENGTH: int = 50000
    URL_FETCH_MIN_TEXT_LENGTH: int = 10

    @field_validator("TRUSTED_PROXIES", mode="before")
    @classmethod
    def parse_trusted_proxies(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [ip.strip() for ip in v.split(",") if ip.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(ip).strip() for ip in v if str(ip).strip()]
        return []

    @field_validator("GEMINI_REQUEST_TIMEOUT_MS")
    @classmethod
    def validate_gemini_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                "GEMINI_REQUEST_TIMEOUT_MS must be a positive integer (milliseconds)"
            )
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
