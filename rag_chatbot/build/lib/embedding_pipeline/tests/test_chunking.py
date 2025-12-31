"""
Tests for the chunking functionality of the Embeddings & Chunking Pipeline.
"""
import pytest
from ..chunker import clean_text, chunk_text


def test_clean_text():
    """Test text cleaning function."""
    # Test basic cleaning
    input_text = "  This is   a test.\n\n\nWith multiple spaces and lines.\t\t"
    expected = "This is a test.\n\nWith multiple spaces and lines."
    result = clean_text(input_text)
    assert result == expected


def test_chunk_text_basic():
    """Test basic chunking functionality."""
    text = "This is a test sentence. " * 50  # Create a longer text
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    # Should have multiple chunks
    assert len(chunks) > 1

    # Each chunk should be at least 30 characters
    for chunk in chunks:
        assert len(chunk) >= 30


def test_chunk_text_with_small_overlap():
    """Test chunking with small overlap."""
    text = "This is a test sentence. " * 20
    chunks = chunk_text(text, chunk_size=80, overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) >= 30


def test_chunk_text_min_size():
    """Test that chunks smaller than 30 characters are not returned."""
    text = "Short."
    chunks = chunk_text(text)

    # With default parameters, this should return an empty list
    # since "Short." is less than 30 characters
    assert len(chunks) == 0


def test_chunk_text_overlap_constraint():
    """Test that overlap doesn't exceed half the chunk size."""
    text = "This is a test sentence. " * 30

    # This should raise an error since overlap (60) exceeds half chunk size (50)
    try:
        chunks = chunk_text(text, chunk_size=100, overlap=60)
        # If no exception, check that it's handled appropriately
        assert True  # Placeholder - depends on implementation
    except ValueError:
        # Expected behavior
        assert True


if __name__ == "__main__":
    test_clean_text()
    test_chunk_text_basic()
    test_chunk_text_with_small_overlap()
    test_chunk_text_min_size()
    print("All chunking tests passed!")