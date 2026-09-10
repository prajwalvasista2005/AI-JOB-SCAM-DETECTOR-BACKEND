from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class InputType(str, Enum):
    TEXT = "text"
    URL = "url"
    IMAGE = "image"


class ScanRequest(BaseModel):
    content: str = Field(
        min_length=10,
        max_length=50000
    )


class ScanResponse(BaseModel):
    id: int
    input_type: InputType
    content: str
    risk_score: int
    verdict: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
