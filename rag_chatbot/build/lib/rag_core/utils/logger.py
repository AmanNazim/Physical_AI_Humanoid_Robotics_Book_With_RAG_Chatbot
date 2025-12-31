import json
import logging
from datetime import datetime
from typing import Any, Dict
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry["extra_data"] = record.extra_data

        return json.dumps(log_entry)


class Logger:
    """JSON logger class for the RAG Chatbot system"""

    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))

        # Avoid adding multiple handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = JSONFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log(self, level: LogLevel, message: str, extra_data: Dict[str, Any] = None):
        """Internal method to log with extra data"""
        if extra_data:
            # Create a custom LogRecord with extra data
            record = self.logger.makeRecord(
                self.logger.name,
                getattr(logging, level.value),
                __file__,
                0,
                message,
                (),
                None
            )
            record.extra_data = extra_data
            self.logger.handle(record)
        else:
            getattr(self.logger, level.lower())(message)

    def debug(self, message: str, extra_data: Dict[str, Any] = None):
        self._log(LogLevel.DEBUG, message, extra_data)

    def info(self, message: str, extra_data: Dict[str, Any] = None):
        self._log(LogLevel.INFO, message, extra_data)

    def warning(self, message: str, extra_data: Dict[str, Any] = None):
        self._log(LogLevel.WARNING, message, extra_data)

    def error(self, message: str, extra_data: Dict[str, Any] = None):
        self._log(LogLevel.ERROR, message, extra_data)

    def critical(self, message: str, extra_data: Dict[str, Any] = None):
        self._log(LogLevel.CRITICAL, message, extra_data)


# Global logger instance
rag_logger = Logger("rag_chatbot")