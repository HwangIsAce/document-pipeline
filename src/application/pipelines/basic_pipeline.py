import cocoindex

from src.infrastructure.operations.file_parser import extract_file_elements
from src.infrastructure.operations.embeddings import TextEmbedder, ImageEmbedder

class BasicPipeline:
    def __init__(
        self,
        text_embedding_model_name: str | None = None,
        image_embedding_model_name: str | None = None,
        qdrant_connection: cocoindex.targets.QdrantConnection | None = None,
        qdrant_collection_text: str | None = None,
        qdrant_collection_image: str | None = None,
    ):
        self.text_embedder = TextEmbedder(text_embedding_model_name)
        self.image_embedder = ImageEmbedder(image_embedding_model_name)
        self.qdrant_connection = qdrant_connection
        self.qdrant_collection_text = qdrant_collection_text
        self.qdrant_collection_image = qdrant_collection_image
    
    @cocoindex.flow_def(name="BasicPipeline")
    def basic_flow(
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
                page["chunks"] = page["text"].transform(
                    cocoindex.functions.SplitRecursively(
                        custom_languages=[
                            cocoindex.functions.CustomLanguageSpec(
                                language_name="text",
                                separators_regex=[
                                    r"\n(\s*\n)+", # 문단 구분
                                    r"[\.!\?]\s+", # 문장 구분
                                    r"\n", # 줄바꿈
                                    r"\s+",
                                ], # 공백 구분
                            )
                        ]
                    ),
                    language="text",
                    chunk_size=600,
                    chunk_overlap=100,
                )
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
                    image["embedding"] = image["data"].transform(self.image_embedder)
                    image_output.collect(
                        id=cocoindex.GeneratedField.UUID,
                        filename=doc["filename"],
                        page=page["page_number"],
                        image_data=image["data"],
                        embedding=image["embedding"],
                    )
                    
        text_output.export(
            "text_output",
            cocoindex.targets.Qdrant(
                connection=self.qdrant_connection,
                collection_name=self.qdrant_collection_text,
            ),
            primary_key_fields=["id"],
        )
        image_output.export(
            "image_embeddings",
            cocoindex.targets.Qdrant(
                connection=self.qdrant_connection,
                collection_name=self.qdrant_collection_image,
            ),
            primary_key_fields=["id"],
        )