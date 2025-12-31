"""
Chunking module for the Embeddings & Chunking Pipeline.
This module implements high-performance, spec-compliant text chunking with overlap handling.
"""
import re
from typing import List
from .config import config
from .tokenizer import count_tokens, encode_tokens, decode_tokens


def clean_text(text: str) -> str:
    """
    Clean and normalize input text according to specifications.

    Args:
        text: Input text to clean

    Returns:
        Cleaned and normalized text
    """
    # Normalize Unicode characters
    text = text.replace('\u00A0', ' ')  # Non-breaking space to regular space
    text = text.replace('\u200B', '')  # Zero-width space removal

    # Remove extra whitespace while preserving line breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple blank lines to double newline

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into chunks with specified size and overlap using token-based chunking.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in tokens (defaults to config value)
        overlap: Overlap size between chunks in tokens (defaults to config value)

    Returns:
        List of text chunks

    Raises:
        ValueError: If chunk parameters are invalid
    """
    if chunk_size is None:
        chunk_size = config.chunk_size
    if overlap is None:
        overlap = config.chunk_overlap

    # Validate parameters
    if chunk_size < 10:
        raise ValueError("Chunk size must be at least 10 tokens")

    if overlap >= chunk_size / 2:
        raise ValueError("Overlap cannot exceed half the chunk size")

    # Clean the input text
    text = clean_text(text)

    # Tokenize the entire text
    tokens = encode_tokens(text)

    # If text is smaller than chunk size, return as single chunk
    if len(tokens) <= chunk_size:
        if len(tokens) >= 10:  # Minimum chunk size requirement
            return [text]
        else:
            return []  # Too small to be a valid chunk

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        # Calculate end position
        end_idx = start_idx + chunk_size

        # If we're near the end of the text, adjust
        if end_idx > len(tokens):
            end_idx = len(tokens)

        # Extract the chunk tokens and decode back to text
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = decode_tokens(chunk_tokens)

        # Ensure chunk meets minimum size requirement
        if len(chunk_tokens) >= 10:
            chunks.append(chunk_text)

        # Move start position by (chunk_size - overlap) for sliding window
        start_idx = end_idx - overlap

        # Handle edge case where we're at the end
        if start_idx >= len(tokens):
            break

    return chunks


def chunk_text_with_semantic_boundaries(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into chunks with respect for semantic boundaries (sentences, paragraphs).
    This is a more advanced chunking approach that tries to respect natural text boundaries.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in tokens (defaults to config value)
        overlap: Overlap size between chunks in tokens (defaults to config value)

    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = config.chunk_size
    if overlap is None:
        overlap = config.chunk_overlap

    # Validate parameters
    if chunk_size < 10:
        raise ValueError("Chunk size must be at least 10 tokens")

    if overlap >= chunk_size / 2:
        raise ValueError("Overlap cannot exceed half the chunk size")

    # Clean the input text
    text = clean_text(text)

    # Split text into sentences to respect sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0

    for sentence in sentences:
        # Count tokens in the sentence
        sentence_tokens = count_tokens(sentence)

        # If adding the next sentence would exceed the chunk size
        if current_chunk_tokens + sentence_tokens > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start a new chunk with overlap consideration
            if overlap > 0:
                # Find a suitable position for overlap
                # We'll need to create an overlap by including some content from the end of the current chunk
                current_chunk = sentence
                current_chunk_tokens = sentence_tokens
            else:
                current_chunk = sentence
                current_chunk_tokens = sentence_tokens
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_chunk_tokens = count_tokens(current_chunk)

        # If current chunk is getting large, add it as a chunk
        if current_chunk_tokens >= chunk_size * 0.9:  # 90% of target size
            chunks.append(current_chunk.strip())
            # Handle overlap for next chunk
            if overlap > 0:
                # Find text that represents the overlap
                all_tokens = encode_tokens(current_chunk)
                overlap_tokens = all_tokens[-overlap:] if len(all_tokens) >= overlap else all_tokens
                current_chunk = decode_tokens(overlap_tokens)
                current_chunk_tokens = len(overlap_tokens)
            else:
                current_chunk = ""
                current_chunk_tokens = 0

    # Add the last chunk if it has content
    if current_chunk.strip() and count_tokens(current_chunk) >= 10:
        chunks.append(current_chunk.strip())

    return chunks