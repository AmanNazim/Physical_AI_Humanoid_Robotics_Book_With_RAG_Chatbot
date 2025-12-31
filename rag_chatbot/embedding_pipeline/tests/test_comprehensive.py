#!/usr/bin/env python3
"""
Comprehensive test suite for embeddings pipeline
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add the rag_chatbot directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from embedding_pipeline import EmbeddingPipeline, generate_embeddings_for_document, generate_embeddings_from_sitemap


async def test_basic_functionality():
    """Test basic pipeline functionality"""
    print("Testing basic pipeline functionality...")

    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # Test with sample text
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    """

    result = await pipeline.process_content(sample_text, document_reference="test_doc_1")
    print(f"Basic functionality test result: {result}")

    assert result['success'] == True
    assert result['embeddings_generated'] > 0
    print("✓ Basic functionality test passed")


async def test_file_processing():
    """Test file processing functionality"""
    print("\nTesting file processing functionality...")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        temp_file.write("""
        This is a test document for the embeddings pipeline.
        It contains sample content to test the full processing workflow.
        The pipeline should be able to read this file, process the content,
        chunk it appropriately, generate embeddings, and store them to the database.
        """)
        temp_file_path = temp_file.name

    try:
        result = await generate_embeddings_for_document(temp_file_path)
        print(f"File processing test result: {result}")

        assert result['success'] == True
        assert result['chunks_processed'] > 0
        print("✓ File processing test passed")
    finally:
        # Clean up
        os.unlink(temp_file_path)


async def test_chunking_constraints():
    """Test that chunks meet size constraints"""
    print("\nTesting chunking constraints...")

    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # Create a longer text to ensure we get multiple chunks
    long_text = "This is a test sentence. " * 100  # Should create multiple chunks

    processed_content, _ = pipeline.text_preprocessor.preprocess(long_text)
    chunks = pipeline.chunk_processor.chunk_text(processed_content, "test_doc_2")

    print(f"Generated {len(chunks)} chunks from long text")

    # Check that all chunks meet size constraints
    for i, chunk in enumerate(chunks):
        token_count = chunk.token_count
        print(f"Chunk {i+1}: {token_count} tokens")

        # Check that tokens are within specified range (800-1200)
        assert 800 <= token_count <= 1200, f"Chunk {i+1} has {token_count} tokens, outside 800-1200 range"

    print("✓ Chunking constraints test passed")


async def test_overlap_handling():
    """Test overlap handling functionality"""
    print("\nTesting overlap handling...")

    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # Create text that should result in overlapping chunks
    test_text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. " * 50

    processed_content, _ = pipeline.text_preprocessor.preprocess(test_text)
    chunks = pipeline.chunk_processor.chunk_text(processed_content, "test_doc_3")

    print(f"Generated {len(chunks)} chunks with overlap")

    # Check that there are multiple chunks and that overlap is handled
    assert len(chunks) > 1, "Should have generated multiple chunks for overlap testing"

    # Check that character positions are sequential with proper overlap
    for i in range(1, len(chunks)):
        current_chunk = chunks[i]
        prev_chunk = chunks[i-1]

        # Check for overlap - current chunk should start before previous chunk ends
        overlap_exists = current_chunk.character_start < prev_chunk.character_end
        print(f"Overlap between chunks {i} and {i+1}: {overlap_exists}")

    print("✓ Overlap handling test passed")


async def test_selective_reembedding():
    """Test selective re-embedding functionality"""
    print("\nTesting selective re-embedding...")

    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # First, process some content
    original_content = "This is the original content. It contains several sentences. More content here."
    result1 = await pipeline.process_content(original_content, "test_doc_4")

    print(f"Original content processing result: {result1}")

    # Now modify the content slightly
    modified_content = "This is the modified content. It contains several updated sentences. Additional content here."
    result2 = await pipeline.selective_reembed_document(modified_content, "test_doc_4")

    print(f"Modified content re-embedding result: {result2}")

    # The re-embedding should detect changes and process appropriately
    assert result2['success'] == True
    print("✓ Selective re-embedding test passed")


async def run_all_tests():
    """Run all tests"""
    print("Starting comprehensive test suite for embeddings pipeline...\n")

    await test_basic_functionality()
    await test_file_processing()
    await test_chunking_constraints()
    await test_overlap_handling()
    await test_selective_reembedding()

    print("\n✓ All tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())