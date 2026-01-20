from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

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


# Extraction related Models
class LlmSpecModel(BaseModel):
    """LLM specification for extraction"""
    api_type: str = Field(..., description="API type: 'openai', 'anthropic', 'custom', etc.")
    model: str = Field(..., description="Model name to use")
    address: Optional[str] = Field(None, description="API endpoint address")
    api_config: Optional[Dict[str, Any]] = Field(None, description="API-specific configuration (e.g., api_key)")


class ExtractionConfig(BaseModel):
    """Configuration for extraction (chunk or image)"""
    fields: List[str] = Field(..., description="List of field names to extract", min_length=1)
    llm_spec: LlmSpecModel = Field(..., description="LLM specification for extraction")
    instruction: Optional[str] = Field(None, description="Optional instruction for extraction")


class IndexExtractionConfig(BaseModel):
    """Extraction configuration for indexing"""
    chunk_extraction: Optional[ExtractionConfig] = Field(None, description="Chunk extraction configuration")
    image_extraction: Optional[ExtractionConfig] = Field(None, description="Image extraction configuration (VLM support pending)")
