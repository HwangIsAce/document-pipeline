import cocoindex
import logging
from typing import Dict, List, Optional

from src.infrastructure.target.targets import create_target
from src.infrastructure.chunkers.chunker import get_chunking_function
from src.infrastructure.parsers.file_parser import extract_file_elements
from src.infrastructure.embedders.text_embedder import TextEmbedder
from src.infrastructure.embedders.image_embedder import ImageEmbedder

logger = logging.getLogger(__name__)

class BasicPipeline:
    _instance: "BasicPipeline | None" = None

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
        
        # Extraction 관련 속성
        self.extraction_llm_spec: Optional[cocoindex.LlmSpec] = None
        self.extraction_fields: List[str] = []
        self.extraction_instruction: Optional[str] = None
        
        BasicPipeline._instance = self
    
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
        """Create a dataclass schema for extraction fields"""
        from dataclasses import make_dataclass
        fields = [(name, str, "") for name in field_names]
        return make_dataclass("ExtractionSchema", fields)
    
    def _should_extract(self) -> bool:
        """Check if extraction is enabled"""
        return (
            self.extraction_llm_spec is not None
            and len(self.extraction_fields) > 0
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
            doc["pages"] = doc["content"].transform(extract_file_elements)
            with doc["pages"].row() as page:
                chunking_func, chunking_kwargs = self._get_chunking_function()
                page["chunks"] = page["text"].transform(chunking_func, **chunking_kwargs)
                
                with page["chunks"].row() as chunk:
                    chunk["embedding"] = chunk["text"].call(self.text_embedder)
                    
                    collect_data = {
                        "id": cocoindex.GeneratedField.UUID,
                        "filename": doc["filename"],
                        "page": page["page_number"],
                        "text": chunk["text"],
                        "embedding": chunk["embedding"],
                    }
                    
                    # LLM extraction 수행
                    if self._should_extract():
                        extraction_schema = self._create_extraction_schema(self.extraction_fields)
                        extract_kwargs = {
                            "llm_spec": self.extraction_llm_spec,
                            "output_type": extraction_schema,
                        }
                        if self.extraction_instruction:
                            extract_kwargs["instruction"] = self.extraction_instruction
                        
                        chunk["extracted"] = chunk["text"].transform(
                            cocoindex.functions.ExtractByLlm(**extract_kwargs)
                        )
                        
                        # extraction 결과를 collect_data에 추가
                        for field_name in self.extraction_fields:
                            collect_data[field_name] = chunk["extracted"][field_name]
                    
                    text_output.collect(**collect_data)
                with page["images"].row() as image:
                    image["embedding"] = image["data"].call(self.image_embedder)
                    
                    collect_data = {
                        "id": cocoindex.GeneratedField.UUID,
                        "filename": doc["filename"],
                        "page": page["page_number"],
                        "image_data": image["data"],
                        "embedding": image["embedding"],
                    }
                    
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