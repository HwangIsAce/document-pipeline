import cocoindex

from typing import Dict

from src.infrastructure.target.targets import create_target
from src.infrastructure.operations.chunking import get_chunking_function
from src.infrastructure.operations.file_parser import extract_file_elements
from src.infrastructure.operations.embeddings import TextEmbedder, ImageEmbedder

class BasicPipeline:
    _instance: "BasicPipeline | None" = None # 클래스 변수로 인스턴스 저장

    def __init__(
        self,
        text_embedding_model_name: str | None = None,
        image_embedding_model_name: str | None = None,
        chunking_method: str = "recursive",
        chunking_kwargs: dict | None = None,
        export_target: str = "qdrant",
        target_kwargs: dict | None = None,
    ):
        self.text_embedder = TextEmbedder(text_embedding_model_name)
        self.image_embedder = ImageEmbedder(image_embedding_model_name)
        
        self.chunking_method = chunking_method
        self.chunking_kwargs = chunking_kwargs or {}
        self.export_target = export_target
        self.target_kwargs = target_kwargs or {}
        
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
                    text_output.collect(
                        id=cocoindex.GeneratedField.UUID,
                        filename=doc["filename"],
                        page=page["page_number"],
                        text=chunk["text"],
                        embedding=chunk["embedding"],
                    )
                with page["images"].row() as image: # image 처리
                    image["embedding"] = image["data"].call(self.image_embedder)
                    image_output.collect(
                        id=cocoindex.GeneratedField.UUID,
                        filename=doc["filename"],
                        page=page["page_number"],
                        image_data=image["data"],
                        embedding=image["embedding"],
                    )
                
        text_target = self._get_export_target(
            self.target_kwargs.get("collection_text", "text_collection"),
            connection=self.target_kwargs.get("connection")
        )
        image_target = self._get_export_target(
            self.target_kwargs.get("collection_image", "image_collection"),
            connection=self.target_kwargs.get("connection")
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