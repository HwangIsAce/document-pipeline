import pytest
from unittest.mock import patch, Mock

from src.infrastructure.target.qdrant import QdrantProvider

class TestQdrantProvider:
    
    def test_get_client(self):
        with patch('src.infrastructure.target.qdrant.QdrantClient') as mock_client:
            provider = QdrantProvider(url="http://localhost:6334")
            provider.get_client()
            mock_client.assert_called_once_with(url="http://localhost:6334", prefer_grpc=True)
            
    def test_get_target(self):
        provider = QdrantProvider(url="http://localhost:6334")
        provider._connection = Mock()
        
        with patch('src.infrastructure.target.qdrant.cocoindex.targets.Qdrant') as mock_target:
            provider.get_target(collection="test-collection")
            mock_target.assert_called_once_with(
                connection=provider._connection,
                collection_name="test-collection"
            )