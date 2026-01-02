import pytest
from unittest.mock import patch

from src.domain.models import PdfPage
from src.infrastructure.parsers.file_parser import UpstageParser

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
            },
            {
                "id": 4,
                "page": 2,
                "category": "paragraph",
                "content": {"text": "", "markdown": "Page 2 markdown text"}
            },
            {
                "id": 5,
                "page": 3,
                "category": "paragraph",
                "content": {"text": "", "markdown": "", "html": "<p>Page 3 html text</p>"}
            }
        ]
        result = parser._group_elements_by_page(elements)
        
        assert 1 in result
        assert 2 in result
        assert 3 in result
        assert "Page 1 text" in result[1]['text_parts'][0]
        assert "Page 2 markdown text" in result[2]['text_parts'][1]  # markdown에서 추출
        assert "Page 3 html text" in result[3]['text_parts'][0]  # html에서 태그 제거 후 추출
        assert len(result[1]['images']) == 1
        
    @patch.dict('os.environ', {'UPSTAGE_API_KEY': 'test_api_key'})
    @patch('src.infrastructure.parsers.file_parser.UpstageParser._call_api')
    def test_extract(self, mock_call_api, mock_upstage_api_response):
        parser = UpstageParser()
        mock_call_api.return_value = mock_upstage_api_response
        
        result = parser.extract(b'test_content')
        
        assert len(result) == 2
        assert result[0].page_number == 1
        assert result[1].page_number == 2
        assert isinstance(result[0], PdfPage)
        
        # 텍스트 추출 확인 (text, markdown, html 모두 포함)
        assert len(result[0].text) > 0  # Page 1에는 text와 markdown이 있음
        assert len(result[1].text) > 0  # Page 2에는 text와 html이 있음
        
        # Markdown에서 추출된 텍스트 확인 (id 5 element가 markdown 사용)
        assert "Markdown content" in result[0].text
        
        # HTML에서 추출된 텍스트 확인 (id 6 element가 html만 있음, 태그 제거 후 추출)
        assert "HTML only text" in result[1].text