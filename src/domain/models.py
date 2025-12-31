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
    
# Enums
class ChunkingMethod(str, Enum):
    RECURSIVE = "recursive"
    # 나중에 추가 가능: SENTENCE = "sentence",...


class ExportTarget(str, Enum):
    QDRANT = "qdrant"
    # 나중에 추가 가능: ELASTICSEARCH = "elasticsearch",...
