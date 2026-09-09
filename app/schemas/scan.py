from enum import Enum
from pydantic import BaseModel, Field


class InputType(str, Enum):
    TEXT = "text"
    URL = "url"


class ScanRequest(BaseModel):
    input_type: InputType
    content: str = Field(
        min_length=10,
        max_length=50000
    )


class ScanResponse(BaseModel):
    input_type: InputType
    content: str