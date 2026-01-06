"""
API routers package for the RAG Chatbot backend.
"""
from . import health, chat, retrieve, embed


# Make routers available for import
from .health import router as health
from .chat import router as chat
from .retrieve import router as retrieve
from .embed import router as embed