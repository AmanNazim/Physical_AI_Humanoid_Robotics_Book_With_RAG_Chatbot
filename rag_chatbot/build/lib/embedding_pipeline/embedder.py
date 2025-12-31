"""
Embedder module for the Embeddings & Chunking Pipeline.
This module implements proper integration with the Gemini Client for generating vector embeddings.
"""
import asyncio
from typing import List
from .config import config
from .gemini_client import get_gemini_client


async def generate_embedding(text: str) -> List[float]:
    """
    Generate a single embedding for the given text using Gemini API.

    Args:
        text: Input text to generate embedding for

    Returns:
        List of float values representing the embedding vector

    Raises:
        ValueError: If embedding dimension doesn't match expected value
        RuntimeError: If API call fails after retries
    """
    client = get_gemini_client()
    return await client.embed_single(text)


async def batch_generate(chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of text chunks using Gemini API with proper batching.

    Args:
        chunks: List of text chunks to generate embeddings for

    Returns:
        List of embedding vectors (each vector is a list of floats)

    Raises:
        ValueError: If any embedding dimension doesn't match expected value
        RuntimeError: If API call fails after retries
    """
    if not chunks:
        return []

    client = get_gemini_client()
    return await client.embed_batch(chunks)