import json
import time
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request

from typing import Optional

from src.application.container import ApplicationContainer
from src.domain.models import ChunkingMethod, ExportTarget
from src.infrastructure.api.schemas import (
    SearchRequest,
    SearchResponse,
    UploadResponse,
    IndexExtractionConfig,
)

logger = logging.getLogger(__name__)

def create_app(container: ApplicationContainer) -> FastAPI:
    """Create FastAPI app with dependency injection"""
    app = FastAPI()
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        logger.info(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} in {process_time:.2f}s")
        return response
    
    @app.post("/index", response_model=UploadResponse)
    async def index_document(
        file: UploadFile = File(...),
        chunking_method: ChunkingMethod = Form(ChunkingMethod.RECURSIVE),
        chunking_kwargs: str = Form("{}"),
        export_target: ExportTarget = Form(ExportTarget.QDRANT),
        target_kwargs: str = Form("{}"),
        extraction_config: Optional[str] = Form(None, description="Extraction configuration as JSON string (IndexExtractionConfig)"),
    ):
        """Index a document. Worker will automatically detect and index the file."""
        try:
            content = await file.read()
            filename = file.filename or "document.pdf"
            
            # Extraction config 파싱 및 검증 (Pydantic 모델 사용)
            extraction_config_model = None
            if extraction_config:
                try:
                    extraction_config_dict = json.loads(extraction_config)
                    extraction_config_model = IndexExtractionConfig(**extraction_config_dict)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid JSON in extraction_config: {str(e)}")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid extraction_config format: {str(e)}")
            
            filename, filepath = await container.index_document(
                file_content=content,
                filename=filename,
                chunking_method=chunking_method.value,
                chunking_kwargs=chunking_kwargs,
                export_target=export_target.value,
                target_kwargs=target_kwargs,
                extraction_config=extraction_config_model,
            )
            
            return UploadResponse(
                message="File uploaded successfully. Indexing will start automatically when worker is running.",
                filename=filename,
                filepath=filepath,
            )
        except Exception as e:
            logger.error(f"Indexing failed: {str(e)}", exc_info=True)
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
            logger.error(f"Search failed: {str(e)}", exc_info=True)
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
            logger.error(f"Image search failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")
    
    return app