import pytest
from unittest.mock import Mock, patch

from src.infrastructure.chunkers.strategies.llm_semantic import (
    llm_semantic_chunk,
    openai_token_count
)

class TestOpenAITokenCount:
    """test openai token count"""
    
    def test_token_count(self):
        text = "Hello world"
        count = openai_token_count(text)
        assert isinstance(count, int)
        assert count > 0

class TestLLMSemanticChunk:
    """test llm semantic chunking"""
    
    @patch('src.infrastructure.chunkers.strategies.llm_semantic.OpenAIClient')
    def test_basic_chunking(self, mock_client_class):
        mock_client = Mock()
        mock_client.create_message.return_value = "split_after: 2"
        mock_client_class.return_value = mock_client
        
        text = "첫 번째 문장. 두 번째 문장. 세 번째 문장. 네 번째 문장."
        
        result = llm_semantic_chunk(
            text=text,
            organization="openai",
            chunk_size=20
        )
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(doc, str) for doc in result)
    
    @patch('src.infrastructure.chunkers.strategies.llm_semantic.OpenAIClient')
    def test_invalid_response_retry(self, mock_client_class):
        """check invalid response retry"""
        mock_client = Mock()
        mock_client.create_message.side_effect = [
            "split_after: 5, 3",  # 오름차순이 아님
            "split_after: 3, 5"   # 유효한 응답
        ]
        mock_client_class.return_value = mock_client
        
        text = "첫 번째. 두 번째. 세 번째. 네 번째. 다섯 번째."
        
        result = llm_semantic_chunk(
            text=text,
            organization="openai",
            chunk_size=10
        )
        
        assert mock_client.create_message.call_count >= 2
        assert isinstance(result, list)
    
    def test_unsupported_organization(self):
        """check unsupported organization error"""
        with pytest.raises(ValueError, match="Unsupported organization"):
            llm_semantic_chunk(
                text="test",
                organization="unsupported"
            )