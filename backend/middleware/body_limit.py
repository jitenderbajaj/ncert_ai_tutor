# FILE: backend/middleware/body_limit.py
"""
Body size limit middleware
"""
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request body size limits"""
    
    def __init__(self, app, max_size: int):
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_size:
                logger.warning(f"Request body too large: {content_length} > {self.max_size}")
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request body too large"}
                )
        
        response = await call_next(request)
        return response
