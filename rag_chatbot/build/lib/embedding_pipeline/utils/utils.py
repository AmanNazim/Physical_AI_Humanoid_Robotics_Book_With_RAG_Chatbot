"""
Utility functions for the Embeddings & Chunking Pipeline.
"""
import hashlib
from typing import List


def generate_content_hash(content: str) -> str:
    """
    Generate SHA-256 hash for content to ensure integrity and detect duplicates.

    Args:
        content: Input content to hash

    Returns:
        SHA-256 hash as hexadecimal string
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def validate_chunk_size(chunk: str, min_size: int = 30) -> bool:
    """
    Validate that a chunk meets minimum size requirements.

    Args:
        chunk: Text chunk to validate
        min_size: Minimum required size in characters

    Returns:
        True if chunk is valid, False otherwise
    """
    return len(chunk) >= min_size


def calculate_token_count(text: str) -> int:
    """
    Calculate approximate token count for text.
    This is a simple estimation using character-based approach.
    For production use, a proper tokenizer should be used.

    Args:
        text: Input text to count tokens for

    Returns:
        Approximate token count
    """
    # Simple estimation: 1 token ~ 4 characters
    return len(text) // 4


def get_file_encoding(file_path: str) -> str:
    """
    Detect file encoding for proper text loading.

    Args:
        file_path: Path to the file

    Returns:
        Detected encoding
    """
    import chardet

    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

        # Handle common encoding issues
        if encoding is None:
            return 'utf-8'

        # Normalize encoding names
        encoding = encoding.lower()
        if encoding in ['ascii', 'utf-8-sig']:
            return 'utf-8'
        elif encoding.startswith('iso-'):
            return 'latin-1'
        else:
            return encoding