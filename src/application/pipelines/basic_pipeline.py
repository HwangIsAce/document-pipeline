import cocoindex
import logging
from dataclasses import make_dataclass, field
from typing import Dict, Optional, List

from src.infrastructure.target.targets import create_target
from src.infrastructure.chunkers.chunker import get_chunking_function
from src.infrastructure.parsers.file_parser import extract_file_elements
from src.infrastructure.embedders.text_embedder import TextEmbedder
from src.infrastructure.embedders.image_embedder import ImageEmbedder

logger = logging.getLogger(__name__)

class BasicPipeline:
    _instance: "BasicPipeline | None" = None # 클래스 변수로 인스턴스 저장

    def __init__(
        self,
        text_embedding_model_name: str | None = None,
        image_embedding_model_name: str | None = None,
        chunking_method: str = "recursive",
        chunking_kwargs: dict | None = None,
        export_target: str | None= None,
        target_kwargs: dict | None = None,
    ):
        self.text_embedder = TextEmbedder(text_embedding_model_name)
        self.image_embedder = ImageEmbedder(image_embedding_model_name)
        
        self.chunking_method = chunking_method
        self.chunking_kwargs = chunking_kwargs or {}
        self.export_target = export_target
        self.target_kwargs = target_kwargs or {}
        
        # Chunk extraction 관련 설정 (기본값: 비활성화)
        self.chunk_extraction_fields: List[str] = []
        self.chunk_extraction_llm_spec: Optional[cocoindex.LlmSpec] = None
        self.chunk_extraction_instruction: Optional[str] = None
        
        # Image extraction 관련 설정 (기본값: 비활성화)
        self.image_extraction_fields: List[str] = []
        self.image_extraction_llm_spec: Optional[cocoindex.LlmSpec] = None
        self.image_extraction_instruction: Optional[str] = None
        
        BasicPipeline._instance = self # 클래스 변수에 인스턴스 저장
    
    @classmethod
    def get_instance(cls) -> "BasicPipeline":
        """Get the pipeline instance"""
        if cls._instance is None:
            raise RuntimeError("BasicPipeline instance not set. Create an instance first.")
        return cls._instance
    
    def _get_chunking_function(self):
        """Get chunking FunctionSpec from registry"""
        return get_chunking_function(
            self.chunking_method,
            **self.chunking_kwargs,
        )
    
    def _create_extraction_schema(self, field_names: List[str]) -> type:
        """Create extraction schema dynamically from field names"""
        fields = [(name, Optional[str], field(default=None)) for name in field_names]
        schema = make_dataclass("ExtractedFields", fields)
        
        # docstring 추가 (LLM이 필드를 이해하기 쉽도록)
        if field_names:
            schema.__doc__ = f"Extracted fields: {', '.join(field_names)}"
        
        return schema
    
    def _should_extract_chunk(self) -> bool:
        """Check if chunk extraction is enabled"""
        return (
            len(self.chunk_extraction_fields) > 0 
            and self.chunk_extraction_llm_spec is not None
        )
    
    def _should_extract_image(self) -> bool:
        """Check if image extraction is enabled"""
        return (
            len(self.image_extraction_fields) > 0 
            and self.image_extraction_llm_spec is not None
        )
        
    def _get_export_target(self, collection_name: str, **extra_kwargs: Dict):
        """Get export TargetSpec from registry with kwargs"""
        target_params = {
            **self.target_kwargs,
            "collection_name": collection_name,
            **extra_kwargs,
        }
        return create_target(
            self.export_target,
            **target_params,
        )
    
    def _basic_flow_impl(
        self,
        flow_builder: cocoindex.FlowBuilder,
        data_scope: cocoindex.DataScope
    ) -> None:
        data_scope["documents"] = flow_builder.add_source(
            cocoindex.sources.LocalFile(path="files", binary=True)
        )

        text_output = data_scope.add_collector()
        image_output = data_scope.add_collector()
        with data_scope["documents"].row() as doc:
            doc["pages"] = doc["content"].transform(extract_file_elements) # upstage API 호출
            with doc["pages"].row() as page:
                chunking_func, chunking_kwargs = self._get_chunking_function()
                page["chunks"] = page["text"].transform(chunking_func, **chunking_kwargs)
                
                with page["chunks"].row() as chunk: # text chunk 처리
                    chunk["embedding"] = chunk["text"].call(self.text_embedder)
                    
                    collect_data = {
                        "id": cocoindex.GeneratedField.UUID,
                        "filename": doc["filename"],
                        "page": page["page_number"],
                        "text": chunk["text"],
                        "embedding": chunk["embedding"],
                    }
                    
                    # Chunk extraction이 활성화된 경우
                    if self._should_extract_chunk():
                        extraction_schema = self._create_extraction_schema(self.chunk_extraction_fields)
                        
                        extract_kwargs = {
                            "llm_spec": self.chunk_extraction_llm_spec,
                            "output_type": extraction_schema,
                        }
                        if self.chunk_extraction_instruction:
                            extract_kwargs["instruction"] = self.chunk_extraction_instruction
                        
                        try:
                            # ExtractByLlm은 transform()을 사용
                            chunk["extracted"] = chunk["text"].transform(
                                cocoindex.functions.ExtractByLlm(**extract_kwargs)
                            )
                            
                            # chunk["extracted"]는 dataclass 인스턴스 (flow context에서 자동 unwrap)
                            for field_name in self.chunk_extraction_fields:
                                field_value = getattr(chunk["extracted"], field_name, None)
                                if field_value is not None:
                                    collect_data[field_name] = field_value
                                    
                        except Exception as e:
                            logger.error(f"Extraction failed for chunk: {e}", exc_info=True)
                            # 에러 발생 시에도 계속 진행 (extraction 없이 저장)
                    
                    text_output.collect(**collect_data)
                with page["images"].row() as image: # image 처리
                    image["embedding"] = image["data"].call(self.image_embedder)
                    
                    collect_data = {
                        "id": cocoindex.GeneratedField.UUID,
                        "filename": doc["filename"],
                        "page": page["page_number"],
                        "image_data": image["data"],
                        "embedding": image["embedding"],
                    }
                    
                    # Image extraction이 활성화된 경우
                    # TODO: ExtractByLlm이 VLM을 지원하면 아래 주석을 해제하고 수정
                    # 현재는 ExtractByLlm이 텍스트만 받으므로 비활성화
                    if self._should_extract_image():
                        # VLM 확장 시: ExtractByLlm이 이미지를 직접 받을 수 있으면
                        # extraction_schema = self._create_extraction_schema(self.image_extraction_fields)
                        # extract_kwargs = {
                        #     "llm_spec": self.image_extraction_llm_spec,
                        #     "output_type": extraction_schema,
                        # }
                        # if self.image_extraction_instruction:
                        #     extract_kwargs["instruction"] = self.image_extraction_instruction
                        # 
                        # image["extracted"] = image["data"].transform(
                        #     cocoindex.functions.ExtractByLlm(**extract_kwargs)
                        # )
                        # 
                        # for field_name in self.image_extraction_fields:
                        #     field_value = getattr(image["extracted"], field_name, None)
                        #     if field_value is not None:
                        #         collect_data[field_name] = field_value
                        
                        # 현재는 비활성화: ExtractByLlm이 VLM을 지원할 때까지 대기
                        pass
                    
                    image_output.collect(**collect_data)
                
        # connection은 Qdrant에만 필요, Chroma는 필요 없음
        extra_kwargs = {}
        if self.export_target == "qdrant":
            extra_kwargs["connection"] = self.target_kwargs.get("connection")
        
        text_target = self._get_export_target(
            self.target_kwargs.get("collection_text", "text_collection"),
            **extra_kwargs
        )
        image_target = self._get_export_target(
            self.target_kwargs.get("collection_image", "image_collection"),
            **extra_kwargs
        )
        
        text_output.export("text_output", text_target, primary_key_fields=["id"])
        image_output.export("image_embeddings", image_target, primary_key_fields=["id"])

@cocoindex.flow_def(name="BasicPipeline")
def basic_flow(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope
) -> None:
    """Flow definition that uses the registered pipeline instance"""
    pipeline = BasicPipeline.get_instance()
    pipeline._basic_flow_impl(flow_builder, data_scope)