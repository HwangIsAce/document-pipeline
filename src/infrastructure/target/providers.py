from typing import Protocol, List, Optional

from src.infrastructure.target.qdrant import QdrantProvider
from src.infrastructure.target.chroma import ChromaProvider
from src.domain.models import SearchResult
from src.domain.exceptions import ValidationError

class VectorDBProvider(Protocol):
    """벡터 DB Provider 인터페이스"""
    def get_client(self):
        """DB 클라이언트 반환"""
        ...
    def get_connection(self):
        """Cocoindex용 connection 반환"""
        ...
    def search(self, collection: str, embedding: List[float], limit: int = 20, score_threshold: Optional[float] = None) -> List[SearchResult]:
        """vector db search"""
        ...

def create_provider(provider_type: str, **kwargs) -> VectorDBProvider:
    """
    Provider Factory
    
    Args:
        provider_type: "qdrant", "weaviate" 등 (EXPORT_TARGET과 동일한 값)
        **kwargs: provider_url 등 Provider별 설정
    
    Returns:
        VectorDBProvider 인스턴스
    """
    if provider_type == "qdrant":
        return QdrantProvider(url=kwargs["provider_url"])
    elif provider_type == "chroma":
        url = kwargs.get("provider_url")
        persist_directory = kwargs.get("persist_directory")
        return ChromaProvider(url=url, persist_directory=persist_directory)
    # elif provider_type == "weaviate":
    #     return WeaviateProvider(url=kwargs["provider_url"])
    else:
        raise ValidationError(
            f"Unknown provider type: {provider_type}. "
            f"Available: ['qdrant', 'chroma']"
        )