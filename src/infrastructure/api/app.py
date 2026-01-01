# app.py
import cocoindex
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from typing import Optional

from src.application.container import ApplicationContainer
from src.domain.models import ChunkingMethod, ExportTarget
from src.infrastructure.api.schemas import (
    SearchRequest,
    SearchResponse,
    UploadResponse,
)

def create_app(container: ApplicationContainer) -> FastAPI:
    """Create FastAPI app with dependency injection"""
    app = FastAPI()
    
    @app.post("/index", response_model=UploadResponse)
    async def index_document(
        file: UploadFile = File(...),
        chunking_method: ChunkingMethod = Form(ChunkingMethod.RECURSIVE),
        chunking_kwargs: str = Form("{}"),
        export_target: ExportTarget = Form(ExportTarget.QDRANT),
        target_kwargs: str = Form("{}"),
        pipeline_kwargs: str = Form("{}"),
    ):
        """Index a document. Worker will automatically detect and index the file."""
        try:
            content = await file.read()
            filename = file.filename or "document.pdf"
            
            filename, filepath = await container.index_document(
                file_content=content,
                filename=filename,
                chunking_method=chunking_method.value,
                chunking_kwargs=chunking_kwargs,
                export_target=export_target.value,
                target_kwargs=target_kwargs,
                pipeline_kwargs=pipeline_kwargs,
            )
            
            return UploadResponse(
                message="File uploaded successfully. Indexing will start automatically when worker is running.",
                filename=filename,
                filepath=filepath,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
    
    @app.post("/search", response_model=SearchResponse)
    async def search_text(request: SearchRequest):
        """Search documents by text query."""
        try:
            results = container.search_service.search(
                query=request.query,
                limit=request.limit,
                score_threshold=request.score_threshold
            )
            return SearchResponse(results=results)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    @app.post("/search/image")
    async def search_image(
        file: UploadFile = File(...),
        limit: int = Query(20, ge=1, le=100),
        score_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    ):
        """Search documents by image."""
        try:
            image_bytes = await file.read()
            results = container.search_service.search_by_image(
                image_bytes=image_bytes,
                limit=limit,
                score_threshold=score_threshold
            )
            return SearchResponse(results=results)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")
    
    return app