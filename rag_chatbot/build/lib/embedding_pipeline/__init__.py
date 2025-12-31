"""
Embeddings & Chunking Pipeline - Main Module

This module provides the complete pipeline for transforming text content into
vector embeddings suitable for semantic search and retrieval in the RAG system.
"""
from .config import config
from .chunker import clean_text, chunk_text
from .embedder import generate_embedding, batch_generate
from .pipeline import process_and_store, process_file
from .utils import generate_content_hash

__all__ = [
    'config',
    'clean_text',
    'chunk_text',
    'generate_embedding',
    'batch_generate',
    'process_and_store',
    'process_file',
    'initialize_qdrant',
    'create_collection_if_not_exists',
    'upsert_embeddings',
    'generate_content_hash'
]
