from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class InputType(str, Enum):
    TEXT = "text"
    URL = "url"
    IMAGE = "image"


class ScanRequest(BaseModel):
    content: str = Field(
        min_length=10,
        max_length=50000
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_whitespace(cls, v: str) -> str:
        non_whitespace_count = sum(1 for c in v if not c.isspace())
        if non_whitespace_count == 0:
            raise ValueError("Content cannot be empty or whitespace only")
        if non_whitespace_count < 10:
            raise ValueError("Content must contain at least 10 non-whitespace characters")
        return v


class ScanResponse(BaseModel):
    id: int
    input_type: InputType
    content: str
    risk_score: int
    verdict: str
    created_at: datetime
    explanation: str | None = None
    indicators: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)
