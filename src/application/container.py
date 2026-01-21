import json
import logging
import cocoindex
from pathlib import Path

from src.infrastructure.target.providers import create_provider
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
        
        # Extraction 기본값 설정 (flow 빌드 시 extraction 구조가 포함되도록)
        self.basic_pipeline.extraction_llm_spec = self.config.EXTRACTION_LLM_SPEC
        self.basic_pipeline.extraction_fields = self.config.EXTRACTION_FIELDS
        self.basic_pipeline.extraction_instruction = self.config.EXTRACTION_INSTRUCTION
        
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
        
        # 파이프라인 설정 업데이트
        final_chunking_method = chunking_method or self.config.CHUNKING_METHOD
        self.basic_pipeline.chunking_method = final_chunking_method
        self.basic_pipeline.chunking_kwargs = merged_chunking_kwargs
        self.basic_pipeline.export_target = final_export_target
        self.basic_pipeline.target_kwargs = merged_target_kwargs
        
        # pipeline_kwargs 처리 (extraction 설정 등)
        user_pipeline_kwargs = self._parse_json(pipeline_kwargs)
        
        if user_pipeline_kwargs:
            # extraction_llm_spec_dict가 있으면 LlmSpec 객체로 변환
            if "extraction_llm_spec_dict" in user_pipeline_kwargs:
                llm_spec_dict = user_pipeline_kwargs.pop("extraction_llm_spec_dict")
                if llm_spec_dict:
                    api_type_str = llm_spec_dict.get("api_type")
                    if api_type_str:
                        api_type = getattr(cocoindex.LlmApiType, api_type_str.upper(), None)
                        if api_type:
                            self.basic_pipeline.extraction_llm_spec = cocoindex.LlmSpec(
                                api_type=api_type,
                                model=llm_spec_dict.get("model"),
                                address=llm_spec_dict.get("address"),
                                api_config=llm_spec_dict.get("api_config"),
                            )
            
            # extraction_fields 설정
            if "extraction_fields" in user_pipeline_kwargs:
                self.basic_pipeline.extraction_fields = user_pipeline_kwargs.pop("extraction_fields", [])
            
            # extraction_instruction 설정
            if "extraction_instruction" in user_pipeline_kwargs:
                self.basic_pipeline.extraction_instruction = user_pipeline_kwargs.pop("extraction_instruction", None)
            
            # 나머지 pipeline_kwargs 설정
            for key, value in user_pipeline_kwargs.items():
                if hasattr(self.basic_pipeline, key):
                    setattr(self.basic_pipeline, key, value)

        logger.info(f"Indexing document '{filename}' to {final_export_target}")

        # 파이프라인 실행
        flow = cocoindex.flow.flow_by_name("BasicPipeline")
        await flow.update_async()
        logger.info(f"Successfully indexed '{filename}'")
        
        return filename, str(filepath)