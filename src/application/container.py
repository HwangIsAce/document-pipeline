import json
import logging
import cocoindex
from pathlib import Path
from typing import Optional

from src.infrastructure.target.providers import create_provider
from src.infrastructure.config import Config
from src.application.pipelines.basic_pipeline import BasicPipeline
from src.application.services.search_service import SearchService
from src.infrastructure.api.schemas import IndexExtractionConfig

logger = logging.getLogger(__name__)

class ApplicationContainer:
    """Application container for indexing pipeline and search service"""    
    def __init__(self, config: Config | None = None):
        self.files_dir = Path("files")
        self.files_dir.mkdir(parents=True, exist_ok=True)

        if config is None:
            config = Config()
        self.config = config
        
        # Provider 타입에 따라 다른 파라미터 사용
        provider_type = self.config.EXPORT_TARGET
        if provider_type == "qdrant":
            provider_url = self.config.TARGET_KWARGS.get("provider_url", "http://localhost:6334")
            self.default_db_provider = create_provider(
                provider_type=provider_type,
                provider_url=provider_url,
            )
        elif provider_type == "chroma":
            url = self.config.TARGET_KWARGS.get("url")
            persist_directory = self.config.TARGET_KWARGS.get("persist_directory", "./chroma_db")
            self.default_db_provider = create_provider(
                provider_type=provider_type,
                provider_url=url,  # create_provider에서 url로 처리
                persist_directory=persist_directory,
            )
        else:
            # 기본값: provider_url 사용
            provider_url = self.config.TARGET_KWARGS.get("provider_url")
            self.default_db_provider = create_provider(
                provider_type=provider_type,
                provider_url=provider_url,
            )
        
        
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
            db_provider=self.default_db_provider,
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
        extraction_config: Optional[IndexExtractionConfig] = None,
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
        
        final_export_target = export_target or self.config.EXPORT_TARGET
        
        user_target_kwargs = self._parse_json(target_kwargs)
        merged_target_kwargs = {
            **self.config.TARGET_KWARGS,
            **user_target_kwargs,
        }
        
        # Qdrant인 경우에만 connection 필요
        if final_export_target == "qdrant":
            provider_url = merged_target_kwargs.get("provider_url", "http://localhost:6334")
            dynamic_provider = create_provider(
                provider_type=final_export_target,
                provider_url=provider_url,
            )
            merged_target_kwargs["connection"] = dynamic_provider.get_connection()
        elif final_export_target == "chroma":
            # Chroma는 connection이 필요 없고, url과 persist_directory를 직접 사용
            # merged_target_kwargs에 이미 포함되어 있음
            pass
        
        collection_text = merged_target_kwargs.get("collection_text", "text_collection")
        collection_image = merged_target_kwargs.get("collection_image", "image_collection")
                
        # 파이프라인 설정 업데이트
        final_chunking_method = chunking_method or self.config.CHUNKING_METHOD
        self.basic_pipeline.chunking_method = final_chunking_method
        self.basic_pipeline.chunking_kwargs = merged_chunking_kwargs
        self.basic_pipeline.export_target = final_export_target
        self.basic_pipeline.target_kwargs = merged_target_kwargs
        
        # Extraction config 처리 (Pydantic 모델 직접 사용)
        if extraction_config is None:
            # extraction_config가 None이면 모든 extraction 설정 초기화
            self.basic_pipeline.chunk_extraction_fields = []
            self.basic_pipeline.chunk_extraction_llm_spec = None
            self.basic_pipeline.chunk_extraction_instruction = None
            self.basic_pipeline.image_extraction_fields = []
            self.basic_pipeline.image_extraction_llm_spec = None
            self.basic_pipeline.image_extraction_instruction = None
        else:
            # Chunk extraction 설정
            if extraction_config.chunk_extraction:
                chunk_ext = extraction_config.chunk_extraction
                self.basic_pipeline.chunk_extraction_fields = chunk_ext.fields
                
                # LLM spec 변환
                api_type_str = chunk_ext.llm_spec.api_type
                if api_type_str:
                    api_type = getattr(cocoindex.LlmApiType, api_type_str.upper(), None)
                    if api_type:
                        self.basic_pipeline.chunk_extraction_llm_spec = cocoindex.LlmSpec(
                            api_type=api_type,
                            model=chunk_ext.llm_spec.model,
                            address=chunk_ext.llm_spec.address,
                            api_config=chunk_ext.llm_spec.api_config,
                        )
                    else:
                        logger.warning(f"Invalid api_type: '{api_type_str}'. Available types: {[e.name for e in cocoindex.LlmApiType]}")
                        self.basic_pipeline.chunk_extraction_llm_spec = None
                else:
                    logger.warning("api_type is empty in chunk_extraction.llm_spec")
                    self.basic_pipeline.chunk_extraction_llm_spec = None
                
                # instruction 설정 (None이어도 명시적으로 설정)
                self.basic_pipeline.chunk_extraction_instruction = chunk_ext.instruction
            else:
                # chunk_extraction이 None이면 초기화
                self.basic_pipeline.chunk_extraction_fields = []
                self.basic_pipeline.chunk_extraction_llm_spec = None
                self.basic_pipeline.chunk_extraction_instruction = None
            
            # Image extraction 설정 (VLM 확장 대비)
            if extraction_config.image_extraction:
                image_ext = extraction_config.image_extraction
                self.basic_pipeline.image_extraction_fields = image_ext.fields
                
                # LLM spec 변환
                api_type_str = image_ext.llm_spec.api_type
                if api_type_str:
                    api_type = getattr(cocoindex.LlmApiType, api_type_str.upper(), None)
                    if api_type:
                        self.basic_pipeline.image_extraction_llm_spec = cocoindex.LlmSpec(
                            api_type=api_type,
                            model=image_ext.llm_spec.model,
                            address=image_ext.llm_spec.address,
                            api_config=image_ext.llm_spec.api_config,
                        )
                    else:
                        self.basic_pipeline.image_extraction_llm_spec = None
                else:
                    self.basic_pipeline.image_extraction_llm_spec = None
                
                # instruction 설정 (None이어도 명시적으로 설정)
                self.basic_pipeline.image_extraction_instruction = image_ext.instruction
            else:
                # image_extraction이 None이면 초기화
                self.basic_pipeline.image_extraction_fields = []
                self.basic_pipeline.image_extraction_llm_spec = None
                self.basic_pipeline.image_extraction_instruction = None

        logger.info(
            f"Indexing document '{filename}' to {final_export_target}: "
            f"text_collection={collection_text}, image_collection={collection_image}"
        )

        # 파이프라인 실행
        try:
            flow = cocoindex.flow.flow_by_name("BasicPipeline")
            await flow.update_async()
            logger.info(f"Successfully saved document '{filename}' to {final_export_target}: text_collection={collection_text}, image_collection={collection_image}")
        except Exception as e:
            logger.error(
                f"Failed to save document '{filename}' to {final_export_target}: {e}",
                exc_info=True
            )
            raise
        
        return filename, str(filepath)