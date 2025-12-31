import os
import cocoindex

class Config:
    def __init__(self):

        self.COCOINDEX_DATABASE_URL = os.getenv("COCOINDEX_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cocoindex_db")
        
        self.CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
        self.TEXT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        self.IMAGE_EMBEDDING_MODEL_NAME = "openai/clip-vit-large-patch14"
        
        self.DB_CONFIGS = {
            "QDRANT": {
                "QDRANT_GRPC_URL": os.getenv("QDRANT_GRPC_URL", "http://localhost:6334"),
                "QDRANT_COLLECTION_TEXT": os.getenv("QDRANT_COLLECTION_TEXT", "text_collection"),
                "QDRANT_COLLECTION_IMAGE": os.getenv("QDRANT_COLLECTION_IMAGE", "image_collection"),
            },
        }
        
        # Chunking 기본값
        self.CHUNKING_METHOD = "recursive"
        self.CHUNKING_KWARGS = {
            "language": "text",
            "chunk_size": 600,
            "chunk_overlap": 100,
        }
        
        # Export 기본값
        self.EXPORT_TARGET = "qdrant"
        self.TARGET_KWARGS = {}