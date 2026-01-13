# FILE: backend/middleware/rate_limit.py
"""
Rate limiting middleware (simple in-memory)
"""
import logging
import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter"""
    
    def __init__(self, app, rpm: int = 60):
        super().__init__(app)
        self.rpm = rpm
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        # Clean old entries
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip]
            if now - ts < 60
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.rpm:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        # Record request
        self.requests[client_ip].append(now)
        
        response = await call_next(request)
        return response
