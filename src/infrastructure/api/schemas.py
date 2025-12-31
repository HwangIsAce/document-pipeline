from pydantic import BaseModel, Field
from typing import Optional

# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=20, ge=1, le=100)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    results: list[dict]


class UploadResponse(BaseModel):
    message: str
    filename: str
    filepath: str
