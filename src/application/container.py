import json
import logging
import cocoindex
from pathlib import Path

from src.infrastructure.target.qdrant import QdrantProvider
from src.infrastructure.config import Config
from src.application.pipelines.basic_pipeline import BasicPipeline
from src.application.services.search_service import SearchService

logger = logging.getLogger(__name__)

class ApplicationContainer:
    """Application container for indexing pipeline and search service"""    
    def __init__(self, config: Config | None = None):
        self.files_dir = Path("files")
        self.files_dir.mkdir(parents=True, exist_ok=True)

        if config is None:
            config = Config()
        self.config = config
        
        # Qdrant Provider 초기화
        qdrant_url = self.config.TARGET_KWARGS["QDRANT_GRPC_URL"]
        self.qdrant_provider = QdrantProvider(url=qdrant_url)
        
        # 인덱싱 파이프라인 초기화
        self.basic_pipeline = BasicPipeline(
            text_embedding_model_name=self.config.TEXT_EMBEDDING_MODEL_NAME,
            image_embedding_model_name=self.config.IMAGE_EMBEDDING_MODEL_NAME,
            chunking_method=self.config.CHUNKING_METHOD,
            chunking_kwargs=self.config.CHUNKING_KWARGS,
            export_target=self.config.EXPORT_TARGET,
            target_kwargs=self.config.TARGET_KWARGS,
        )
        
        # Search Service 초기화
        self.search_service = SearchService(
            qdrant_provider=self.qdrant_provider,
            text_embedder=self.basic_pipeline.text_embedder,
            image_embedder=self.basic_pipeline.image_embedder,
            text_collection_name=self.config.TARGET_KWARGS["collection_text"],
            image_collection_name=self.config.TARGET_KWARGS["collection_image"],
        )
    
    @staticmethod
    def _parse_json(value: str) -> dict:
        """Parse JSON string, return empty dict on error"""
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    
    async def index_document(
        self,
        file_content: bytes,
        filename: str,
        chunking_method: str | None = None,
        chunking_kwargs: str = "{}",
        export_target: str | None = None,
        target_kwargs: str = "{}",
        pipeline_kwargs: str = "{}",
    ) -> tuple[str, str]:
        """Index a document. Returns (filename, filepath)"""
        # 파일 저장
        filepath = self.files_dir / filename
        with open(filepath, "wb") as f:
            f.write(file_content)
        
        # config에서 기본값 로드하고 사용자 입력과 병합
        user_chunking_kwargs = self._parse_json(chunking_kwargs)
        merged_chunking_kwargs = {
            **self.config.CHUNKING_KWARGS,
            **user_chunking_kwargs,
        }
        
        user_target_kwargs = self._parse_json(target_kwargs)
        merged_target_kwargs = {
            **self.config.TARGET_KWARGS,
            "connection": self.qdrant_provider.get_connection(),
            **user_target_kwargs,
        }
        
        collection_text = merged_target_kwargs.get("collection_text", "text_collection")
        collection_image = merged_target_kwargs.get("collection_image", "image_collection")
        
        # 파이프라인 설정 업데이트
        if chunking_method:
            self.basic_pipeline.chunking_method = chunking_method
        self.basic_pipeline.chunking_kwargs = merged_chunking_kwargs
        if export_target:
            self.basic_pipeline.export_target = export_target
        self.basic_pipeline.target_kwargs = merged_target_kwargs
        
        # pipeline_kwargs 처리
        user_pipeline_kwargs = self._parse_json(pipeline_kwargs)
        if user_pipeline_kwargs:
            for key, value in user_pipeline_kwargs.items():
                if hasattr(self.basic_pipeline, key):
                    setattr(self.basic_pipeline, key, value)
        
        logger.info(f"Indexing document '{filename}' to Qdrant: text_collection={collection_text}, image_collection={collection_image}")

        # 파이프라인 실행
        try:
            flow = cocoindex.flow.flow_by_name("BasicPipeline")
            logger.debug("Flow retrieved, starting update_async()")
            await flow.update_async()
            logger.info(f"Successfully saved document '{filename}' to Qdrant: text_collection={collection_text}, image_collection={collection_image}")
        except Exception as e:
            logger.error(
                f"Failed to save document '{filename}' to Qdrant: {e}",
                exc_info=True
            )
            raise
        
        return filename, str(filepath)