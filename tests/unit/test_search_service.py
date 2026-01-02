import pytest
from unittest.mock import Mock, patch

from src.infrastructure.target.qdrant import QdrantProvider
from src.infrastructure.embedders.text_embedder import TextEmbedder
from src.infrastructure.embedders.image_embedder import ImageEmbedder
from src.application.services.search_service import SearchService

@pytest.fixture
def search_service():
    provider = QdrantProvider(url="http://localhost:6334")
    mock_client = Mock()
    provider.get_client = Mock(return_value=mock_client)
    
    text_embedder = TextEmbedder(text_embedding_model_name=None)
    image_embedder = Mock(spec=ImageEmbedder)
    
    return SearchService(
        qdrant_provider=provider,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        text_collection_name="test-text-collection",
        image_collection_name="test-image-collection",
    )
    
class TestSearchService:
    def test_normalize_scores(self, search_service):
        results = [
            {"id": "1", "score": 0.1},
            {"id": "2", "score": 0.9},
        ]
        normalized = search_service._normalize_scores(results)
        
        assert len(normalized) == 2
        assert normalized[0]["normalized_score"] == 0.0
        assert normalized[1]["normalized_score"] == 1.0
    
    def test_search_combines_text_and_image_results(self, search_service):
        text_result = Mock()
        text_result.id = "t1"
        text_result.score = 0.9
        text_result.payload = {"text": "test"}
        
        image_result = Mock()
        image_result.id = "i1"
        image_result.score = 0.8
        image_result.payload = {"image": "test.png"}
        
        search_service.client.search.side_effect = [
            [text_result],  # text collection
            [image_result],  # image collection
        ]
        
        with patch.object(search_service, '_embed_text_for_search', return_value=[0.1, 0.2]):
            with patch.object(search_service, '_embed_text_with_clip', return_value=[0.3, 0.4]):
                results = search_service.search("test", limit=10)
        
        assert len(results) == 2
        assert any(r["collection_type"] == "text" for r in results)
        assert any(r["collection_type"] == "image" for r in results)
        assert all("normalized_score" in r for r in results)
        scores = [r["normalized_score"] for r in results]
        assert scores == sorted(scores, reverse=True)