import os
import cocoindex

class Config:
    def __init__(self):


        
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
