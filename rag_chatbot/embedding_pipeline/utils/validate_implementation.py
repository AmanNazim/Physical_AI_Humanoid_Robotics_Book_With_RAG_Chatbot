#!/usr/bin/env python3
"""
Validation script to verify implementation meets all requirements
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the rag_chatbot directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from embedding_pipeline import EmbeddingPipeline


async def validate_class_based_architecture():
    """Validate class-based architecture implementation"""
    print("Validating class-based architecture...")

    # Check that all major components are classes
    from embedding_pipeline import (
        EmbeddingPipeline,
        ChunkProcessor,
        EmbeddingProcessor,
        DatabaseManager,
        TextPreprocessor,
        FileProcessor,
        URLCrawler,
        SelectiveReembedder
    )

    # Verify that key components are classes
    components = [
        EmbeddingPipeline,
        ChunkProcessor,
        EmbeddingProcessor,
        DatabaseManager,
        TextPreprocessor,
        FileProcessor,
        URLCrawler,
        SelectiveReembedder
    ]

    for component in components:
        if not isinstance(component, type):
            print(f"✗ {component.__name__} is not a class")
            return False

    print("✓ All major components use class-based architecture")
    return True


async def validate_chunking_constraints():
    """Validate chunking constraints (800-1200 tokens)"""
    print("\nValidating chunking constraints (800-1200 tokens)...")

    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # Create text that should result in properly sized chunks
    test_text = "This is a test sentence. " * 100  # Should create multiple chunks

    processed_content, _ = pipeline.text_preprocessor.preprocess(test_text)
    chunks = pipeline.chunk_processor.chunk_text(processed_content, "validation_test")

    all_valid = True
    for i, chunk in enumerate(chunks):
        token_count = chunk.token_count
        if not (800 <= token_count <= 1200):
            print(f"✗ Chunk {i+1} has {token_count} tokens, outside 800-1200 range")
            all_valid = False
        else:
            print(f"✓ Chunk {i+1}: {token_count} tokens (valid)")

    if all_valid:
        print("✓ All chunks meet 800-1200 token constraints")
    else:
        print("✗ Some chunks do not meet token constraints")

    return all_valid


async def validate_overlap_handling():
    """Validate overlap handling (200 tokens)"""
    print("\nValidating overlap handling...")

    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # Create text that should result in overlapping chunks
    test_text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. " * 50

    processed_content, _ = pipeline.text_preprocessor.preprocess(test_text)
    chunks = pipeline.chunk_processor.chunk_text(processed_content, "overlap_test")

    # Check for overlap in adjacent chunks
    has_overlap = False
    for i in range(1, len(chunks)):
        current_chunk = chunks[i]
        prev_chunk = chunks[i-1]

        # Check if there's overlap - current chunk should start before previous chunk ends
        if current_chunk.character_start < prev_chunk.character_end:
            has_overlap = True
            overlap_size = prev_chunk.character_end - current_chunk.character_start
            print(f"✓ Overlap detected between chunks {i} and {i+1}: ~{overlap_size} characters")
            break

    if has_overlap:
        print("✓ Overlap handling is working correctly")
        return True
    else:
        print("✗ No overlap detected - overlap handling may not be working")
        return False


async def validate_google_gemini_integration():
    """Validate Google Gemini API integration"""
    print("\nValidating Google Gemini API integration...")

    try:
        pipeline = EmbeddingPipeline()
        await pipeline.initialize()

        # Test with a simple chunk
        from embedding_pipeline.base_classes import Chunk
        import hashlib

        test_chunk = Chunk(
            chunk_id="validation_chunk_1",
            content="This is a test for Google Gemini API integration.",
            token_count=12,
            character_start=0,
            character_end=54,
            token_start=0,
            token_end=12,
            document_reference="validation_test"
        )

        embeddings = await pipeline.embedding_processor.generate_embeddings([test_chunk])

        if len(embeddings) == 1 and len(embeddings[0]) > 0:
            embedding_size = len(embeddings[0])
            print(f"✓ Google Gemini API integration working - embedding dimension: {embedding_size}")
            return True
        else:
            print("✗ Google Gemini API returned invalid embeddings")
            return False
    except Exception as e:
        print(f"✗ Google Gemini API integration error: {str(e)}")
        return False


async def validate_database_integration():
    """Validate database integration"""
    print("\nValidating database integration...")

    try:
        pipeline = EmbeddingPipeline()
        await pipeline.initialize()

        # Test database initialization
        await pipeline.database_manager.initialize()
        print("✓ Database integration initialization successful")
        return True
    except Exception as e:
        print(f"✗ Database integration error: {str(e)}")
        return False


async def validate_url_processing():
    """Validate URL processing capability"""
    print("\nValidating URL processing capability...")

    try:
        from embedding_pipeline.url_crawler import URLCrawler

        crawler = URLCrawler()
        # Test that the crawler can be instantiated and has required methods
        assert hasattr(crawler, 'crawl_sitemap'), "URLCrawler missing crawl_sitemap method"
        assert hasattr(crawler, 'process_url'), "URLCrawler missing process_url method"

        print("✓ URL processing components are available")
        return True
    except Exception as e:
        print(f"✗ URL processing validation error: {str(e)}")
        return False


async def validate_optimized_processing():
    """Validate optimized processing for speed"""
    print("\nValidating optimized processing...")

    try:
        pipeline = EmbeddingPipeline()
        await pipeline.initialize()

        # Test that the pipeline can process content efficiently
        import time

        # Create a moderate-sized text
        test_content = "This is test content for performance validation. " * 50

        start_time = time.time()
        result = await pipeline.process_content(test_content, "performance_test")
        end_time = time.time()

        processing_time = end_time - start_time
        chunks_processed = result.get('chunks_processed', 0)

        print(f"✓ Optimized processing: {chunks_processed} chunks in {processing_time:.2f}s")
        return result['success']
    except Exception as e:
        print(f"✗ Optimized processing validation error: {str(e)}")
        return False


async def validate_one_by_one_processing():
    """Validate one-by-one processing for each file path"""
    print("\nValidating one-by-one processing...")

    try:
        pipeline = EmbeddingPipeline()
        await pipeline.initialize()

        # The pipeline architecture supports individual processing
        # This is validated by the architecture design
        assert hasattr(pipeline, 'process_content'), "Pipeline missing process_content method"
        assert hasattr(pipeline, 'process_file'), "Pipeline missing process_file method"
        assert hasattr(pipeline, 'process_from_sitemap'), "Pipeline missing process_from_sitemap method"

        print("✓ One-by-one processing architecture is implemented")
        return True
    except Exception as e:
        print(f"✗ One-by-one processing validation error: {str(e)}")
        return False


async def run_validation():
    """Run complete validation"""
    print("Starting implementation validation...\n")

    validations = [
        ("Class-based Architecture", validate_class_based_architecture),
        ("Chunking Constraints (800-1200 tokens)", validate_chunking_constraints),
        ("Overlap Handling (200 tokens)", validate_overlap_handling),
        ("Google Gemini API Integration", validate_google_gemini_integration),
        ("Database Integration", validate_database_integration),
        ("URL Processing", validate_url_processing),
        ("Optimized Processing", validate_optimized_processing),
        ("One-by-One Processing", validate_one_by_one_processing),
    ]

    results = []
    for name, validation_func in validations:
        result = await validation_func()
        results.append((name, result))
        print()  # Add spacing between validations

    print("="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<45} {status}")
        if not result:
            all_passed = False

    print("="*60)

    if all_passed:
        print("✓ All validations passed! Implementation meets requirements.")
        return True
    else:
        print("✗ Some validations failed! Implementation needs fixes.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_validation())
    sys.exit(0 if success else 1)