import pytest
from unittest.mock import patch

from src.domain.models import PdfPage
from src.infrastructure.operations.file_parser import UpstageParser

class TestUpstageParser:
    
    @patch.dict('os.environ', {'UPSTAGE_API_KEY': 'test_api_key'})
    def test_is_image_category(self):
        parser = UpstageParser()
        assert parser._is_image_category('chart') == True
        assert parser._is_image_category('table') == True
        assert parser._is_image_category('paragraph') == False
        
    @patch.dict('os.environ', {'UPSTAGE_API_KEY': 'test_api_key'})
    def test_group_elements_by_page(self, sample_base64_image):
        parser = UpstageParser()
        
        elements = [
            {
                "id": 1,
                "page": 1,
                "category": "paragraph",
                "content": {"text": "Page 1 text"}
            },
            {
                "id": 2,
                "page": 1,
                "category": "chart",
                "base64_encoding": sample_base64_image,
                "content": {"text": ""}
            },
            {
                "id": 3,
                "page": 2,
                "category": "paragraph",
                "content": {"text": "Page 2 text"}
            }
        ]
        result = parser._group_elements_by_page(elements)
        
        assert 1 in result
        assert 2 in result
        assert "Page 1 text" in result[1]['text_parts'][0]
        assert len(result[1]['images']) == 1
        
    @patch.dict('os.environ', {'UPSTAGE_API_KEY': 'test_api_key'})
    @patch('src.infrastructure.operations.file_parser.UpstageParser._call_api')
    def test_extract(self, mock_call_api, mock_upstage_api_response):
        parser = UpstageParser()
        mock_call_api.return_value = mock_upstage_api_response
        
        result = parser.extract(b'test_content')
        
        assert len(result) == 2
        assert result[0].page_number == 1
        assert result[1].page_number == 2
        assert isinstance(result[0], PdfPage)