from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class InputType(str, Enum):
    TEXT = "text"
    URL = "url"
    IMAGE = "image"


class AnalysisMode(str, Enum):
    AI_ASSISTED = "ai_assisted"
    RULES_ONLY = "rules_only"
    LEGACY_UNKNOWN = "legacy_unknown"


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


class JobDetails(BaseModel):
    company_name: str | None = None
    job_title: str | None = None
    salary: str | None = None
    location: str | None = None
    employment_type: str | None = None
    experience_required: str | None = None
    education_required: str | None = None
    application_method: str | None = None
    application_url: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ScanResponse(BaseModel):
    id: int
    input_type: InputType
    content: str
    risk_score: int
    verdict: str
    analysis_mode: AnalysisMode = AnalysisMode.LEGACY_UNKNOWN
    created_at: datetime
    explanation: str | None = None
    indicators: list[str] | None = None
    job_details: JobDetails | None = None

    @field_validator("analysis_mode", mode="before")
    @classmethod
    def default_legacy_unknown(cls, v: Any) -> Any:
        if v is None:
            return AnalysisMode.LEGACY_UNKNOWN
        return v

    model_config = ConfigDict(from_attributes=True)
