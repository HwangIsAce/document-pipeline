class DocumentPipelineError(Exception):
    """Base exception for all pipeline errors"""
    status_code = 500
    
    def __init__(self, message: str = "Internal server error"):
        self.message = message
        super().__init__(self.message)


class ValidationError(DocumentPipelineError):
    """Invalid input data (file format, JSON, etc.)"""
    status_code = 400
    
    def __init__(self, message: str = "Invalid input data"):
        super().__init__(message)


class NotFoundError(DocumentPipelineError):
    """Resource not found (collection, document, etc.)"""
    status_code = 404
    
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message)


class ExternalServiceError(DocumentPipelineError):
    """External service failed (Upstage API, Vector DB, etc.)"""
    status_code = 502
    
    def __init__(self, message: str = "External service error"):
        super().__init__(message)
