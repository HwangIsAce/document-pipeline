"""Chroma Vector DB Integration

이 파일은 Chroma 관련 모든 구현을 포함합니다:
- ChromaProvider: 검색용 (SearchService에서 사용)
- ChromaTarget: 인덱싱용 TargetSpec (CocoIndex에서 사용)
- ChromaTargetConnector: 인덱싱용 Connector (CocoIndex에서 사용)
"""

from typing import List, Optional, Any
import chromadb

import cocoindex
from cocoindex import op

from src.domain.models import SearchResult


# ============================================================================
# Provider: 검색용 (SearchService에서 사용)
# ============================================================================

class ChromaProvider:
    """Chroma vector DB Provider"""
    
    def __init__(self, url: Optional[str] = None, persist_directory: Optional[str] = None):
        self.url = url
        self.persist_directory = persist_directory or "./chroma_db"
        
        if url:
            self._client = chromadb.HttpClient(host=url.split("://")[1].split(":")[0] if "://" in url else url)
        else:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
    
    def get_client(self):
        """Return Chroma client"""
        return self._client
    
    def get_connection(self):
        """Return connection (Chroma doesn't need connection)"""
        return None
    
    def search(
        self,
        collection: str,
        embedding: List[float],
        limit: int = 20,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Chroma vector DB search"""
        try:
            chroma_collection = self._client.get_or_create_collection(name=collection)
            results = chroma_collection.query(
                query_embeddings=[embedding],
                n_results=limit
            )
            
            search_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    score = 1.0 - distance
                    
                    if score_threshold is not None and score < score_threshold:
                        continue
                    
                    metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                    search_results.append(
                        SearchResult(
                            id=str(doc_id),
                            score=score,
                            payload=metadata,
                            collection_type=""
                        )
                    )
            
            return search_results
        except Exception:
            return []


# ============================================================================
# Target & Connector: 인덱싱용 (CocoIndex에서 사용)
# ============================================================================

class ChromaTarget(op.TargetSpec):
    """Chroma target configuration for CocoIndex"""
    collection_name: str
    persist_directory: Optional[str] = None
    url: Optional[str] = None


@op.target_connector(spec_cls=ChromaTarget)
class ChromaTargetConnector:
    """Chroma target connector for CocoIndex"""
    
    @staticmethod
    def get_persistent_key(spec: ChromaTarget, target_name: str) -> str:
        """Return a unique identifier for this target instance"""
        return f"chroma_{spec.collection_name}"
    
    @staticmethod
    def apply_setup_change(
        key: str,
        previous: ChromaTarget | None,
        current: ChromaTarget | None
    ) -> None:
        """Apply setup changes (create/update/delete collection)"""
        if current is None:
            return
        
        if current.url:
            client = chromadb.HttpClient(host=current.url.split("://")[1].split(":")[0] if "://" in current.url else current.url)
        else:
            persist_dir = current.persist_directory or "./chroma_db"
            client = chromadb.PersistentClient(path=persist_dir)
        
        try:
            client.get_or_create_collection(name=current.collection_name)
        except Exception:
            pass
    
    @staticmethod
    def mutate(
        *all_mutations: tuple[ChromaTarget, dict],
    ) -> None:
        """Apply data mutations (upsert/delete) to Chroma"""
        for spec, mutations in all_mutations:
            if spec.url:
                client = chromadb.HttpClient(host=spec.url.split("://")[1].split(":")[0] if "://" in spec.url else spec.url)
            else:
                persist_dir = spec.persist_directory or "./chroma_db"
                client = chromadb.PersistentClient(path=persist_dir)
            
            collection = client.get_or_create_collection(name=spec.collection_name)
            
            ids_to_upsert = []
            embeddings_to_upsert = []
            metadatas_to_upsert = []
            ids_to_delete = []
            
            for doc_id, mutation in mutations.items():
                doc_id_str = str(doc_id)
                
                if mutation is None:
                    ids_to_delete.append(doc_id_str)
                else:
                    ids_to_upsert.append(doc_id_str)
                    embedding = mutation.get("embedding", [])
                    metadata = {k: v for k, v in mutation.items() if k != "embedding"}
                    if "image_data" in metadata:
                        del metadata["image_data"]
                    embeddings_to_upsert.append(embedding)
                    metadatas_to_upsert.append(metadata)
            
            if ids_to_upsert:
                collection.upsert(
                    ids=ids_to_upsert,
                    embeddings=embeddings_to_upsert,
                    metadatas=metadatas_to_upsert,
                )
            
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
    
    @staticmethod
    def describe(key: str) -> str:
        """Return human-readable description"""
        return f"Chroma target: {key}"

