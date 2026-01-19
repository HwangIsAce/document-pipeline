from enum import Enum
from dataclasses import dataclass

@dataclass
class PdfImage:
    name: str
    data: bytes
    
@dataclass
class PdfPage:
    page_number: int
    text: str
    images: list[PdfImage]
    
@dataclass
class TextChunk:
    """Text chunk structure matching SplitRecursively output"""
    text: str
    
# Enums
class ChunkingMethod(str, Enum):
    RECURSIVE = "recursive"
    LLM_SEMANTIC = "llm_semantic"

class ExportTarget(str, Enum):
    QDRANT = "qdrant"
    CHROMA = "chroma"
    # 나중에 추가 가능: ELASTICSEARCH = "elasticsearch",...
    
@dataclass
class SearchResult:
    """표준화된 검색 결과 모델"""
    id: str
    score: float
    payload: dict
    collection_type: str
    
