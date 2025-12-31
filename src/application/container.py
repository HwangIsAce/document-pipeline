from src.infrastructure.target.qdrant import QdrantProvider
from src.infrastructure.config import Config
from src.application.pipelines.basic_pipeline import BasicPipeline
from src.application.services.search_service import SearchService

class ApplicationContainer:
    """Application container for indexing pipeline and search service"""    
    def __init__(self, config: Config | None = None):

        if config is None:
            config = Config()
        self.config = config
        
        # Qdrant Provider 초기화
        qdrant_url = self.config.DB_CONFIGS["QDRANT"]["QDRANT_GRPC_URL"]
        self.qdrant_provider = QdrantProvider(url=qdrant_url)
        
        # 인덱싱 파이프라인 초기화
        self.basic_pipeline = BasicPipeline(
            text_embedding_model_name=self.config.TEXT_EMBEDDING_MODEL_NAME,
            image_embedding_model_name=self.config.IMAGE_EMBEDDING_MODEL_NAME,
            qdrant_connection=self.qdrant_provider.get_connection(),
            qdrant_collection_text=self.config.DB_CONFIGS["QDRANT"]["QDRANT_COLLECTION_TEXT"],
            qdrant_collection_image=self.config.DB_CONFIGS["QDRANT"]["QDRANT_COLLECTION_IMAGE"],
        )
        
        # Search Service 초기화
        self.search_service = SearchService(
            qdrant_provider=self.qdrant_provider,
            text_embedder=self.basic_pipeline.text_embedder,
            image_embedder=self.basic_pipeline.image_embedder,
            text_collection_name=self.config.DB_CONFIGS["QDRANT"]["QDRANT_COLLECTION_TEXT"],
            image_collection_name=self.config.DB_CONFIGS["QDRANT"]["QDRANT_COLLECTION_IMAGE"],
        )