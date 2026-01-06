"""
Rate limiting middleware for the RAG Chatbot API.
"""
import time
from collections import defaultdict, deque
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from ..utils.logger import rag_logger
from typing import Dict


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent abuse of the API.
    Implements a sliding window rate limiter.
    """

    def __init__(self, app: FastAPI, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # Get client IP address
        client_ip = request.client.host

        # Get current time
        current_time = time.time()

        # Remove requests older than 1 minute
        while (self.requests[client_ip] and
               current_time - self.requests[client_ip][0] > 60):
            self.requests[client_ip].popleft()

        # Check if rate limit is exceeded
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            rag_logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed"
                }
            )

        # Add current request timestamp
        self.requests[client_ip].append(current_time)

        # Process the request
        response = await call_next(request)
        return response


def add_rate_limit_middleware(app: FastAPI, requests_per_minute: int = 60):
    """
    Add rate limiting middleware to the FastAPI application.

    Args:
        app: FastAPI application instance
        requests_per_minute: Number of requests allowed per minute per IP
    """
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=requests_per_minute
    )