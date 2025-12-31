"""
Embeddings & Chunking Pipeline for Global RAG Chatbot System
Package initialization
"""
from .config import *
from .base_classes import *
from .url_crawler import *
from .file_processor import *
from .text_preprocessor import *
from .chunking_engine import *
from .gemini_client import *
from .database import *
from .reembedding import *
from .pipeline import *

__version__ = "1.0.0"
__author__ = "Global RAG Chatbot System"