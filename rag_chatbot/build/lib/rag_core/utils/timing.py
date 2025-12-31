import time
import asyncio
from functools import wraps
from typing import Callable, Any
from .logger import rag_logger


def timing_decorator(func):
    """
    Decorator to time function execution and log the duration.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            rag_logger.info(
                f"Async function {func.__name__} executed in {duration:.4f} seconds",
                extra_data={
                    "function": func.__name__,
                    "duration_seconds": duration,
                    "args_count": len(args)
                }
            )

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            rag_logger.info(
                f"Function {func.__name__} executed in {duration:.4f} seconds",
                extra_data={
                    "function": func.__name__,
                    "duration_seconds": duration,
                    "args_count": len(args)
                }
            )

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class Timer:
    """
    Context manager for timing code blocks.
    """
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        rag_logger.info(
            f"Timer '{self.name}' completed in {duration:.4f} seconds",
            extra_data={
                "timer_name": self.name,
                "duration_seconds": duration
            }
        )

    @property
    def duration(self) -> float:
        """Get the duration in seconds. Returns 0 if not finished."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return 0.0


def time_async_operation(operation_name: str = "async_operation"):
    """
    Decorator factory to time async operations with a custom name.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time
                rag_logger.info(
                    f"Async operation '{operation_name}' completed in {duration:.4f} seconds",
                    extra_data={
                        "operation_name": operation_name,
                        "duration_seconds": duration,
                        "function": func.__name__
                    }
                )
        return wrapper
    return decorator


def time_sync_operation(operation_name: str = "sync_operation"):
    """
    Decorator factory to time sync operations with a custom name.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time
                rag_logger.info(
                    f"Operation '{operation_name}' completed in {duration:.4f} seconds",
                    extra_data={
                        "operation_name": operation_name,
                        "duration_seconds": duration,
                        "function": func.__name__
                    }
                )
        return wrapper
    return decorator