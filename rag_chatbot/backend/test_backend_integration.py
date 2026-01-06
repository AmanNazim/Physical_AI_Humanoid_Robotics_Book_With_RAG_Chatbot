#!/usr/bin/env python3
"""
Test script to verify integration with existing subsystems.
This script tests that the FastAPI backend can properly integrate
with the existing database and embedding subsystems.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from databases.database_manager import database_manager
from embedding_pipeline.pipeline import EmbeddingPipeline
from embedding_pipeline.gemini_client import EmbeddingProcessor
from backend.services.retrieval_service import RetrievalService
from backend.services.rag_service import RAGService
from backend.services.embedding_service import EmbeddingService
from backend.utils.logger import rag_logger


async def test_database_integration():
    """Test integration with the database subsystem."""
    print("Testing database integration...")

    try:
        # Initialize the database manager
        await database_manager.connect_all()
        print("✓ Database manager connected successfully")

        # Test that we can access both Qdrant and PostgreSQL
        qdrant_client = database_manager.qdrant()
        postgres_client = database_manager.postgres()

        print("✓ Successfully accessed Qdrant client")
        print("✓ Successfully accessed PostgreSQL client")

        # Test basic health check
        postgres_health = await postgres_client.health_check()
        print(f"✓ PostgreSQL health check: {postgres_health}")

        return True
    except Exception as e:
        print(f"✗ Database integration test failed: {str(e)}")
        return False
    finally:
        try:
            await database_manager.close_all()
            print("✓ Database connections closed")
        except:
            pass


async def test_embedding_integration():
    """Test integration with the embedding pipeline."""
    print("\nTesting embedding pipeline integration...")

    try:
        # Initialize the embedding pipeline
        pipeline = EmbeddingPipeline()
        await pipeline.initialize()
        print("✓ Embedding pipeline initialized successfully")

        # Test the embedding processor
        processor = EmbeddingProcessor()
        await processor.initialize()
        print("✓ Embedding processor initialized successfully")

        # Test generating embeddings for a simple text
        from embedding_pipeline.base_classes import Chunk

        test_chunk = Chunk(
            id="test_chunk",
            content="This is a test document for integration testing.",
            document_reference="test_doc",
            metadata={"test": True}
        )

        embeddings = await processor.generate_embeddings([test_chunk])

        if embeddings and len(embeddings) > 0:
            print(f"✓ Successfully generated embeddings: {len(embeddings[0])} dimensions")
        else:
            print("✗ Failed to generate embeddings")
            return False

        return True
    except Exception as e:
        print(f"✗ Embedding integration test failed: {str(e)}")
        return False


async def test_backend_services():
    """Test the backend services integration."""
    print("\nTesting backend services integration...")

    try:
        # Test retrieval service
        retrieval_service = RetrievalService()
        await retrieval_service.initialize()
        print("✓ Retrieval service initialized successfully")

        # Test RAG service
        rag_service = RAGService()
        await rag_service.initialize()
        print("✓ RAG service initialized successfully")

        # Test embedding service
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        print("✓ Embedding service initialized successfully")

        return True
    except Exception as e:
        print(f"✗ Backend services integration test failed: {str(e)}")
        return False


async def test_end_to_end_flow():
    """Test an end-to-end flow if possible."""
    print("\nTesting end-to-end flow...")

    try:
        # Initialize services
        retrieval_service = RetrievalService()
        await retrieval_service.initialize()

        # Test query validation
        is_valid = await retrieval_service.validate_query("What is artificial intelligence?")
        if is_valid:
            print("✓ Query validation working")
        else:
            print("✗ Query validation failed")
            return False

        # Try to retrieve with a simple query (this may return no results if no data is indexed)
        sources = await retrieval_service.retrieve_by_query(
            query="What is artificial intelligence?",
            top_k=3
        )
        print(f"✓ Retrieval service working, found {len(sources)} sources")

        # Test RAG service
        rag_service = RAGService()
        await rag_service.initialize()

        result = await rag_service.generate_response(
            query="What is artificial intelligence?",
            top_k=3
        )

        print(f"✓ RAG service working, answer length: {len(result.get('answer', ''))}")

        return True
    except Exception as e:
        print(f"✗ End-to-end flow test failed: {str(e)}")
        return False


async def main():
    """Run all integration tests."""
    print("Starting integration tests for FastAPI Backend Subsystem...\n")

    tests = [
        ("Database Integration", test_database_integration),
        ("Embedding Integration", test_embedding_integration),
        ("Backend Services", test_backend_services),
        ("End-to-End Flow", test_end_to_end_flow),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)

        success = await test_func()
        results.append((test_name, success))

    print(f"\n{'='*50}")
    print("Integration Test Results:")
    print('='*50)

    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False

    print(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)