import pytest
from unittest.mock import Mock, MagicMock
import base64
import io
from PIL import Image

@pytest.fixture
def sample_image_bytes():
    """테스트용 이미지 바이트 생성"""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

@pytest.fixture
def sample_base64_image():
    """테스트용 base64 인코딩된 이미지"""
    img = Image.new('RGB', (100, 100), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')

@pytest.fixture
def mock_upstage_api_response(sample_base64_image):
    """Upstage API 응답 mock (실제 응답 구조 반영)"""
    # 실제 응답과 유사한 구조로 mock 데이터 생성
    # sample_base64_image를 실제 chart 이미지로 사용
    chart_base64 = sample_base64_image
    
    return {
        "api": "2.0",
        "model": "document-parse",
        "elements": [
            {
                "id": 0,
                "page": 1,
                "category": "header",
                "content": {
                    "html": "<header id='0' style='font-size:16px'>Test Document Title</header>",
                    "markdown": "",
                    "text": "Test Document Title"
                },
                "coordinates": [
                    {"x": 0.0305, "y": 0.0349},
                    {"x": 0.4957, "y": 0.0349},
                    {"x": 0.4957, "y": 0.054},
                    {"x": 0.0305, "y": 0.054}
                ]
            },
            {
                "id": 1,
                "page": 1,
                "category": "paragraph",
                "content": {
                    "html": "<p id='1'>Sample paragraph text content</p>",
                    "markdown": "",
                    "text": "Sample paragraph text content"
                },
                "coordinates": [
                    {"x": 0.0339, "y": 0.081},
                    {"x": 0.9385, "y": 0.081},
                    {"x": 0.9385, "y": 0.1321},
                    {"x": 0.0339, "y": 0.1321}
                ]
            },
            {
                "id": 2,
                "page": 1,
                "category": "chart",
                "base64_encoding": chart_base64,
                "content": {
                    "html": "<figure id='2' data-category='chart'><img /></figure>",
                    "markdown": "",
                    "text": ""
                },
                "coordinates": [
                    {"x": 0.0381, "y": 0.148},
                    {"x": 0.9348, "y": 0.148},
                    {"x": 0.9348, "y": 0.3913},
                    {"x": 0.0381, "y": 0.3913}
                ]
            },
            {
                "id": 3,
                "page": 2,
                "category": "paragraph",
                "content": {
                    "html": "<p id='3'>Page 2 text content</p>",
                    "markdown": "",
                    "text": "Page 2 text content"
                },
                "coordinates": [
                    {"x": 0.0358, "y": 0.4432},
                    {"x": 0.9429, "y": 0.4432},
                    {"x": 0.9429, "y": 0.6864},
                    {"x": 0.0358, "y": 0.6864}
                ]
            },
            {
                "id": 4,
                "page": 2,
                "category": "table",
                "base64_encoding": chart_base64,  # 테스트용으로 동일한 이미지 사용
                "content": {
                    "html": "<table id='4'>...</table>",
                    "markdown": "",
                    "text": ""
                },
                "coordinates": [
                    {"x": 0.05, "y": 0.2},
                    {"x": 0.9, "y": 0.2},
                    {"x": 0.9, "y": 0.5},
                    {"x": 0.05, "y": 0.5}
                ]
            },
            {
                "id": 5,
                "page": 1,
                "category": "paragraph",
                "content": {
                    "html": "<p id='5'>Another paragraph</p>",
                    "markdown": "# Markdown content",  # text가 없으면 markdown 사용
                    "text": ""
                },
                "coordinates": [
                    {"x": 0.0353, "y": 0.7028},
                    {"x": 0.9424, "y": 0.7028},
                    {"x": 0.9424, "y": 0.808},
                    {"x": 0.0353, "y": 0.808}
                ]
            },
            {
                "id": 6,
                "page": 2,
                "category": "paragraph",
                "content": {
                    "html": "<p id='6'>HTML only text</p>",
                    "markdown": "",  # text와 markdown이 모두 비어있으면 html 사용
                    "text": ""
                },
                "coordinates": [
                    {"x": 0.0353, "y": 0.81},
                    {"x": 0.9424, "y": 0.81},
                    {"x": 0.9424, "y": 0.91},
                    {"x": 0.0353, "y": 0.91}
                ]
            }
        ],
        "content": {
            "html": "...",
            "markdown": "...",
            "text": "..."
        },
        "usage": {
            "pages": 2
        }
    }