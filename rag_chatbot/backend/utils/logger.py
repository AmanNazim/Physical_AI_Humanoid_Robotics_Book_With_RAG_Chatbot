"""
Logging utilities for the RAG Chatbot API.
"""
import logging
import sys
import os

# Add the project root to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.config import settings


# Create a custom logger
rag_logger = logging.getLogger(settings.app_name)
rag_logger.setLevel(getattr(logging, settings.log_level.upper()))

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, settings.log_level.upper()))

# Create formatters and add it to handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)

# Add handlers to the logger
rag_logger.addHandler(console_handler)

# Prevent propagation to avoid duplicate logs
rag_logger.propagate = False