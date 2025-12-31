#!/usr/bin/env python3
"""
Health check script for embeddings pipeline
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the rag_chatbot directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from embedding_pipeline import EmbeddingPipeline


async def check_config():
    """Check configuration"""
    print("Checking configuration...")
    try:
        from embedding_pipeline.config import validate_config
        validate_config()
        print("✓ Configuration is valid")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {str(e)}")
        return False


async def check_pipeline_initialization():
    """Check pipeline initialization"""
    print("\nChecking pipeline initialization...")
    try:
        pipeline = EmbeddingPipeline()
        await pipeline.initialize()
        print("✓ Pipeline initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Pipeline initialization error: {str(e)}")
        return False


async def check_gemini_api():
    """Check Google Gemini API connectivity"""
    print("\nChecking Google Gemini API connectivity...")
    try:
        pipeline = EmbeddingPipeline()
        await pipeline.initialize()

        # Test with a simple chunk
        from embedding_pipeline.base_classes import Chunk
        import hashlib

        test_chunk = Chunk(
            chunk_id="test_chunk_1",
            content="This is a test for API connectivity.",
            token_count=10,
            character_start=0,
            character_end=42,
            token_start=0,
            token_end=10,
            document_reference="health_check"
        )

        embeddings = await pipeline.embedding_processor.generate_embeddings([test_chunk])

        if len(embeddings) > 0 and len(embeddings[0]) > 0:
            print("✓ Google Gemini API connectivity OK")
            return True
        else:
            print("✗ Google Gemini API returned empty embeddings")
            return False
    except Exception as e:
        print(f"✗ Google Gemini API error: {str(e)}")
        return False


async def check_database_connectivity():
    """Check database connectivity"""
    print("\nChecking database connectivity...")
    try:
        pipeline = EmbeddingPipeline()
        await pipeline.initialize()

        # Test database initialization
        await pipeline.database_manager.initialize()
        print("✓ Database connectivity OK")
        return True
    except Exception as e:
        print(f"✗ Database connectivity error: {str(e)}")
        return False


async def check_basic_processing():
    """Check basic processing functionality"""
    print("\nChecking basic processing functionality...")
    try:
        pipeline = EmbeddingPipeline()
        await pipeline.initialize()

        test_content = "This is a health check test. The embeddings pipeline is functioning properly."
        result = await pipeline.process_content(test_content, "health_check_doc")

        if result['success'] and result['embeddings_generated'] > 0:
            print("✓ Basic processing functionality OK")
            return True
        else:
            print(f"✗ Basic processing failed: {result}")
            return False
    except Exception as e:
        print(f"✗ Basic processing error: {str(e)}")
        return False


async def run_health_check():
    """Run complete health check"""
    print("Starting embeddings pipeline health check...\n")

    checks = [
        ("Configuration", check_config),
        ("Pipeline Initialization", check_pipeline_initialization),
        ("Google Gemini API", check_gemini_api),
        ("Database Connectivity", check_database_connectivity),
        ("Basic Processing", check_basic_processing),
    ]

    results = []
    for name, check_func in checks:
        result = await check_func()
        results.append((name, result))

    print("\n" + "="*50)
    print("HEALTH CHECK RESULTS")
    print("="*50)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<30} {status}")
        if not result:
            all_passed = False

    print("="*50)

    if all_passed:
        print("✓ All health checks passed!")
        return True
    else:
        print("✗ Some health checks failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_health_check())
    sys.exit(0 if success else 1)