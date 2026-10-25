
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
