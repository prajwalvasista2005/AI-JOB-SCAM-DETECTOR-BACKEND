
import ipaddress
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

    # Explicit list of trusted proxy IP addresses or CIDR ranges.
    # If empty, X-Forwarded-For is ignored and direct client IP is used.
    TRUSTED_PROXIES: list[str] = []

    # Database Connection Pool Settings
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 5
    DB_POOL_TIMEOUT_SECONDS: float = 15.0
    DB_POOL_RECYCLE_SECONDS: int = 1800  # Default 30m; deployment-configurable
    DB_POOL_PRE_PING: bool = True

    # Scan Concurrency Admission Settings
    SCAN_MAX_CONCURRENCY: int = 10
    SCAN_CONCURRENCY_TIMEOUT_SECONDS: float = 5.0

    # Rate Limiting Settings
    REGISTER_RATE_LIMIT_MAX_REQUESTS: int = 5
    REGISTER_RATE_LIMIT_WINDOW_SECONDS: int = 60
    SCAN_RATE_LIMIT_MAX_REQUESTS: int = 10
    SCAN_RATE_LIMIT_WINDOW_SECONDS: int = 60
    SCAN_HISTORY_RATE_LIMIT_MAX_REQUESTS: int = 30
    SCAN_HISTORY_RATE_LIMIT_WINDOW_SECONDS: int = 60

    # Configurable allowed origins for CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

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
    def parse_and_validate_trusted_proxies(cls, v: Any) -> list[str]:
        # Allow None, empty string, or empty list as valid 'no proxies configured' (direct client)
        if v is None:
            return []
        if isinstance(v, str):
            if not v:
                return []
            if not v.strip():
                raise ValueError("TRUSTED_PROXIES cannot be a whitespace-only string")
            raw_items = v.split(",")
        elif isinstance(v, (list, tuple, set)):
            if len(v) == 0:
                return []
            raw_items = list(v)
        else:
            raise ValueError(f"Unsupported type for TRUSTED_PROXIES: {type(v).__name__}")

        validated: list[str] = []
        for raw in raw_items:
            if not isinstance(raw, str):
                raise ValueError(f"Elements in TRUSTED_PROXIES must be strings, got {type(raw).__name__}")
            cleaned = raw.strip()
            if not cleaned:
                # Reject empty segments between commas, e.g. "10.0.0.1,,10.0.0.2"
                raise ValueError("Empty or whitespace proxy entry found in TRUSTED_PROXIES")
            try:
                if "/" in cleaned:
                    ipaddress.ip_network(cleaned, strict=False)
                else:
                    ipaddress.ip_address(cleaned)
            except ValueError as exc:
                raise ValueError(f"Invalid IP address or CIDR in TRUSTED_PROXIES: '{cleaned}'") from exc
            validated.append(cleaned)
        return validated

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        raw_items: list[Any]
        if isinstance(v, str):
            raw_items = v.split(",")
        elif isinstance(v, (list, tuple, set)):
            raw_items = list(v)
        else:
            return []

        cleaned: list[str] = []
        for item in raw_items:
            s = str(item).strip().rstrip("/")
            if s and s not in cleaned:
                cleaned.append(s)
        return cleaned

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
