import os
import io
import base64
import re
import requests
import logging
import cocoindex
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

from typing import Dict, List

from src.domain.models import PdfImage, PdfPage
from src.domain.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)

IMG_THUMBNAIL_SIZE = (512, 512)

class UpstageParser:
    
    def __init__(self):
        self.api_key = None
        self._prepare()
    
    def _prepare(self) -> None:
        """initialize environment variables"""
        BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
        load_dotenv(BASE_DIR / ".env", override=True)
        
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ExternalServiceError("UPSTAGE_API_KEY environment variable is required")
    
    def _call_api(self, content: bytes) -> Dict:
        """call upstage API"""
        url = "https://api.upstage.ai/v1/document-digitization"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"document": ("document", content)}
        data = {
            "ocr": "force",
            "base64_encoding": "['table', 'chart', 'figure']",
            "model": "document-parse"
        }
        
        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            raise ExternalServiceError(f"Upstage API failed: {e}")   
    
    def _is_image_category(self, category: str) -> bool:
        """check if the category is an image category"""
        image_categories = ["chart", "table", "figure"]
        return category in image_categories
    
    def _process_image(self, image_data: str, image_id: int) -> PdfImage | None:
        """process base64 encoded image"""
        try: 
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))
            
            if img.width < 16 or img.height < 16:
                return None
            
            thumbnail = io.BytesIO()
            img.thumbnail(IMG_THUMBNAIL_SIZE)
            img.save(thumbnail, img.format or "PNG")
            
            return PdfImage(
                name=f"image_{image_id}.png", 
                data=thumbnail.getvalue()
            )
        
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            return None
            
    def _group_elements_by_page(self, elements: List) -> Dict:
        """group elements by page"""
        pages_dict = defaultdict(lambda: {"text_parts": [], "images": []})
        
        for element in elements:
            page_num = element.get("page", 1)
            category = element.get("category", "")
            content_data = element.get("content", {})
            element_id = element.get("id", 0)
            
            # 텍스트 추출 - text 우선, 없으면 markdown, 마지막으로 html 사용
            text = content_data.get("text", "")
            if not text or not text.strip():
                text = content_data.get("markdown", "")
            if not text or not text.strip():
                html_text = content_data.get("html", "")
                # HTML에서 텍스트 추출 (간단한 방법: 태그 제거)
                if html_text:
                    import re
                    # HTML 태그 제거
                    text = re.sub(r'<[^>]+>', '', html_text)
            
            if text and text.strip():
                pages_dict[page_num]["text_parts"].append(text.strip())
            
            if self._is_image_category(category):
                image_data = element.get("base64_encoding", "")
                if image_data:
                    pdf_image = self._process_image(image_data, element_id)
                    if pdf_image:
                        pages_dict[page_num]["images"].append(pdf_image)
        
        return pages_dict
            
    def extract(self, content: bytes) -> List[PdfPage]:
        """extract texts and images from a PDF file"""
        result = self._call_api(content)
        elements = result.get("elements", [])
        pages_dict = self._group_elements_by_page(elements)
        
        result_pages = []
        for page_num in sorted(pages_dict.keys()):
            page_data = pages_dict[page_num]
            text = " ".join(page_data["text_parts"])
            images = page_data["images"]
            
            # 로깅 추가: 텍스트 추출 확인
            if not text.strip():
                logger.warning(f"Page {page_num}: No text extracted (text_parts count: {len(page_data['text_parts'])})")
            else:
                logger.debug(f"Page {page_num}: Extracted {len(text)} characters of text, {len(images)} images")
            
            result_pages.append(PdfPage(
                page_number=page_num, 
                text=text, 
                images=images
            ))
            
        return result_pages
    
@cocoindex.op.function()
def extract_file_elements(content: bytes) -> list[PdfPage]:
    """wrapper function for extract_pdf_elements"""
    extractor = UpstageParser()
    return extractor.extract(content)
        