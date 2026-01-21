import os
import cocoindex

class Config:
    def __init__(self):

        self.COCOINDEX_DATABASE_URL = os.getenv("COCOINDEX_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cocoindex_db")
        
        self.CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
        self.TEXT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        self.IMAGE_EMBEDDING_MODEL_NAME = "openai/clip-vit-large-patch14"
        
        
        # Recursive Chunking 기본값
        self.CHUNKING_METHOD = "recursive"
        self.CHUNKING_KWARGS = {
            "language": "text",
            "chunk_size": 600,
            "chunk_overlap": 100,
        }
        
        # # LLM Semantic Chunking 기본값
        # self.CHUNKING_METHOD = "llm_semantic"
        # self.CHUNKING_KWARGS = {
        #     "organization": "openai",
        #     "api_key": None,
        #     "model_name": "gpt-4o-mini",
        #     "chunk_size": 50,
        #     "chunk_overlap": 10,
        # }
    
        # # Qdrant 기본값
        # self.EXPORT_TARGET = "qdrant"
        # self.TARGET_KWARGS = {
        #     "provider_url": "http://localhost:6334",
        #     "collection_text": "text_collection",
        #     "collection_image": "image_collection",
        # }
        
        # Chroma 기본값
        self.EXPORT_TARGET = "chroma"
        self.TARGET_KWARGS = {
            "url": None,  # None이면 로컬 모드
            "persist_directory": "./chroma_db",
            "collection_text": "text_collection",
            "collection_image": "image_collection",
        }
        
        # Extraction 기본값 (flow 빌드 시 extraction 구조가 포함되도록 설정)
        # 실제 extraction은 API 요청 시 extraction_llm_spec이 설정된 경우에만 수행됨
        self.EXTRACTION_FIELDS = ["summary", "key_concepts"]  # 기본 extraction 필드
        self.EXTRACTION_LLM_SPEC = cocoindex.LlmSpec(
            api_type=cocoindex.LlmApiType.OPENAI,
            model="gpt-4o-mini",
        )
        self.EXTRACTION_INSTRUCTION = "Extract a brief summary and key concepts from the text"