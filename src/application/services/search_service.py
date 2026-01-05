from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer

from src.infrastructure.target.providers import VectorDBProvider
from src.infrastructure.embedders.text_embedder import TextEmbedder
from src.infrastructure.embedders.image_embedder import ImageEmbedder


class SearchService:
    """Search service for vector database"""    
    def __init__(
        self,
        db_provider: VectorDBProvider,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder,
        text_collection_name: str,
        image_collection_name: str,
    ):
        self.client = db_provider.get_client()
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.text_collection = text_collection_name
        self.image_collection = image_collection_name
        
        # SentenceTransformer 모델 (한 번만 생성)
        model_name = self.text_embedder.text_embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._text_model = SentenceTransformer(model_name)
    
    def _embed_text_for_search(self, text: str) -> List[float]:
        """Embed text with SentenceTransformer model for text collection search"""
        embedding = self._text_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def _embed_text_with_clip(self, text: str) -> List[float]:
        """Embed text with CLIP model for image collection search"""
        model, processor = self.image_embedder._get_clip_model()
        inputs = processor(text=text, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        return features[0].tolist()
    
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """normalize scores to 0-1 range"""
        if not results or len(results) == 1:
            return [{**r, "normalized_score": r["score"]} for r in results]
        
        scores = [r["score"] for r in results]
        min_score, max_score = min(scores), max(scores)
        
        if max_score == min_score:
            return [{**r, "normalized_score": 1.0} for r in results]
        
        return [
            {**r, "normalized_score": (r["score"] - min_score) / (max_score - min_score)}
            for r in results
        ]
    
    def search(
        self, 
        query: str, 
        limit: int = 20, 
        score_threshold: float | None = None
    ) -> List[Dict[str, Any]]:
        """Search collections with text query"""
        text_embedding = self._embed_text_for_search(query)
        text_results = [
            {"id": r.id, "score": r.score, "payload": r.payload, "collection_type": "text"}
            for r in self.client.search(self.text_collection, text_embedding, limit=limit, score_threshold=score_threshold)
        ]
        
        # 수정: _embed_text_with_clip 사용
        image_embedding = self._embed_text_with_clip(query)
        image_results = [
            {"id": r.id, "score": r.score, "payload": r.payload, "collection_type": "image"}
            for r in self.client.search(self.image_collection, image_embedding, limit=limit, score_threshold=score_threshold)
        ]
        
        normalized_text = self._normalize_scores(text_results)
        normalized_image = self._normalize_scores(image_results)
        
        combined_results = normalized_text + normalized_image
        combined_results.sort(key=lambda x: x["normalized_score"], reverse=True)
        return combined_results
    
    def search_by_image(
        self, 
        image_bytes: bytes, 
        limit: int = 20, 
        score_threshold: float | None = None
    ) -> List[Dict[str, Any]]:
        """Search image collection with image bytes"""
        image_embedding = self.image_embedder(image_bytes)
        
        results = self.client.search(
            self.image_collection,
            image_embedding,
            limit=limit,
            score_threshold=score_threshold,
        )
        
        return [
            {"id": r.id, "score": r.score, "payload": r.payload, "collection_type": "image"}
            for r in results
        ]