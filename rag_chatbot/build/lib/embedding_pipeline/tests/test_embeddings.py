"""
Tests for the embedding functionality of the Embeddings & Chunking Pipeline.
"""
import pytest
from ..config import config


def test_embedding_dimension():
    """Test that the configured embedding dimension is correct."""
    assert config.embedding_dim == 1024, f"Embedding dimension should be 1024 as per specification, got {config.embedding_dim}"


def test_config_parameters():
    """Test that required config parameters are set."""
    # Test that required parameters exist
    assert hasattr(config, 'embed_model_name')
    assert hasattr(config, 'chunk_size')
    assert hasattr(config, 'chunk_overlap')
    assert hasattr(config, 'embedding_dim')
    assert hasattr(config, 'qdrant_host')
    assert hasattr(config, 'qdrant_collection_name')
    assert hasattr(config, 'max_batch_size')


def test_chunk_size_constraints():
    """Test that chunk size parameters are within specification."""
    assert config.min_chunk_size == 800, "Minimum chunk size should be 800 tokens"
    assert config.max_chunk_size == 1200, "Maximum chunk size should be 1200 tokens"
    assert config.chunk_size == 1000, "Default chunk size should be 1000 tokens"


def test_batch_size_constraint():
    """Test that batch size is appropriate for embedding API."""
    # As per specification, the API supports up to 96 chunks per batch
    assert config.max_batch_size == 96, f"Max batch size should be 96 as per API limits, got {config.max_batch_size}"


if __name__ == "__main__":
    test_embedding_dimension()
    test_config_parameters()
    test_chunk_size_constraints()
    test_batch_size_constraint()
    print("All embedding tests passed!")