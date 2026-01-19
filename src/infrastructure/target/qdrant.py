import cocoindex

from qdrant_client import QdrantClient

from typing import List, Optional

from src.domain.models import SearchResult

class QdrantProvider:
    def __init__(self, url):
        self.url = url
        self._connection_name = "qdrant_connection"
        self._connection = None
        self._vector_names_cache = {}  # 컬렉션별 벡터 이름 캐시

    def get_connection(self):
        """get connection object for use in Qdrant target"""
        if self._connection is None:
            try:
                self._connection = cocoindex.add_auth_entry(
                    self._connection_name,
                    cocoindex.targets.QdrantConnection(grpc_url=self.url)
                )
            except RuntimeError as e:
                if "already exists" in str(e):
                    self._connection = cocoindex.targets.QdrantConnection(grpc_url=self.url)
                else:
                    raise
        return self._connection

    def get_client(self):
        """get database client information"""
        return QdrantClient(url=self.url, prefer_grpc=True)
    
    def get_target(self, collection: str):
        """get database collection information"""
        return cocoindex.targets.Qdrant(
            connection=self.get_connection(),
            collection_name=collection
        )
        
    def _get_vector_name(self, client: QdrantClient, collection: str) -> Optional[str]:
        """컬렉션의 벡터 이름 확인 (named vectors인 경우)"""
        if collection in self._vector_names_cache:
            return self._vector_names_cache[collection]
        
        try:
            vectors_config = client.get_collection(collection).config.params.vectors
            if isinstance(vectors_config, dict) and vectors_config:
                vector_name = list(vectors_config.keys())[0]
                self._vector_names_cache[collection] = vector_name
                return vector_name
        except Exception:
            pass
        
        self._vector_names_cache[collection] = None
        return None
    
    def search(
        self, 
        collection: str, 
        embedding: List[float], 
        limit: int = 20, 
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Qdrant vector search"""
        client = self.get_client()
        vector_name = self._get_vector_name(client, collection)
        
        query_params = {
            "collection_name": collection,
            "query": embedding,
            "limit": limit,
            "score_threshold": score_threshold,
            "with_payload": True,
            "with_vectors": False
        }
        if vector_name:
            query_params["using"] = vector_name
        
        response = client.query_points(**query_params)
        
        return [
            SearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload or {},
                collection_type=""
            )
            for point in response.points
        ]
