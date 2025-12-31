from dataclasses import dataclass

@dataclass
class PdfImage:
    name: str
    data: bytes
    
@dataclass
class PdfPage:
    page_number: int
    text: str
    images: list[PdfImage]