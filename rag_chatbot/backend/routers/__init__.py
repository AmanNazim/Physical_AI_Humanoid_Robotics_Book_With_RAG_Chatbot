"""
API routers package for the RAG Chatbot backend.
"""
from . import health, chat, retrieve, embed, config


# Make routers available for import
from .health import router as health
from .chat import router as chat
from .retrieve import router as retrieve
from .embed import router as embed
from .config import router as config