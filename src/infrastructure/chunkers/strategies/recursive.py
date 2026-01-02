import re
from typing import List

DEFAULT_SEPARATORS = [
    r"\n(\s*\n)+",      # 문단 구분
    r"[\.!\?]\s+",      # 문장 구분
    r"\n",              # 단일 줄바꿈
    r"\s+",             # 공백
]

def get_recursive_separators() -> List[str]:
    """Get default recursive chunking separators"""
    return DEFAULT_SEPARATORS.copy()

def recursive_chunk_text(
    text: str, 
    separators: List[str] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int = 0
) -> List[str]:
    """Chunk text using recursive strategy with given separators"""
    if separators is None:
        separators = DEFAULT_SEPARATORS
    
    if not text.strip():
        return []
    
    for separator in separators:
        parts = re.split(separator, text)
        if len(parts) > 1:
            chunks = []
            for part in parts:
                if part.strip():
                    if chunk_size and len(part) > chunk_size: # chunk_size가 지정되어 있고 초과하면 재귀적으로 더 작게 나누기
                        sub_chunks = recursive_chunk_text(
                            part, 
                            separators=separators[1:],  # 다음 구분자로 재귀적으로 더 작게 나누기
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(part.strip())
            
            if chunks:
                return chunks
    
    return [text.strip()] if text.strip() else [] # 구분자로 나눌 수 없으면 그대로 반환
