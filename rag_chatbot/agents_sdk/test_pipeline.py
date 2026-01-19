#!/usr/bin/env python3
"""
Test script to verify the complete RAG pipeline using all subsystems except ChatKit.
This will test the full flow: query -> retrieval -> intelligence service -> response
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.services.rag_service import RAGService
from agents_sdk.services.intelligence_service import IntelligenceService
from backend.services.retrieval_service import RetrievalService
from shared.config import settings


async def test_question_answer():
    """
    Test asking a specific question and getting an answer from the RAG system.
    """
    print("\n" + "="*50)
    print("❓ Question & Answer Test")
    print("="*50)

    # Specific question to test
    question = "What is ROS 2?"

    print(f"Asking: '{question}'")

    try:
        # Initialize the RAG service
        rag_service = RAGService()
        await rag_service.initialize()

        # Get the answer
        result = await rag_service.generate_response(
            query=question,
            top_k=5
        )

        answer = result.get('answer', 'No answer generated')
        print(f"\n🎯 Answer: {answer}")

        # Print sources used if available
        if 'sources' in result and result['sources']:
            print(f"\n🔗 Sources used: {len(result['sources'])}")

        return answer

    except Exception as e:
        print(f"❌ Error getting answer: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_rag_pipeline():
    """
    Test the complete RAG pipeline with a sample question about the book.
    """
    print("🧪 Starting RAG Pipeline Test...")
    print("="*50)

    # Test query about the book
    test_query = "What is ROS 2?"

    print(f"❓ Query: {test_query}")
    print("-" * 30)

    try:
        # Initialize the RAG service
        print("🔧 Initializing RAG Service...")
        rag_service = RAGService()
        await rag_service.initialize()
        print("✅ RAG Service initialized successfully")

        # Test the retrieval service directly
        print("\n🔍 Testing Retrieval Service...")
        retrieval_service = RetrievalService()
        await retrieval_service.initialize()

        # Retrieve relevant context
        sources = await retrieval_service.retrieve_by_query(
            query=test_query,
            top_k=5
        )

        print(f"📚 Retrieved {len(sources)} sources:")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. Score: {source.score:.3f}")
            print(f"     Text snippet: {source.text[:100]}...")
            print(f"     Chunk ID: {source.chunk_id}")
            print()

        if not sources:
            print("❌ No sources retrieved - this indicates the embeddings might not be properly stored")
            return

        # Test the intelligence service
        print("🧠 Testing Intelligence Service...")
        intelligence_service = IntelligenceService()
        await intelligence_service.initialize()

        # Process the query with retrieved context
        result = await intelligence_service.process_query(
            user_query=test_query,
            context_chunks=sources
        )

        print(f"🎯 Response: {result.get('text', 'No response generated')}")
        print()

        # Test the full RAG service (which should now use IntelligenceService)
        print("🔄 Testing Full RAG Service Pipeline...")
        rag_result = await rag_service.generate_response(
            query=test_query,
            top_k=5
        )

        print(f"💬 Final RAG Response: {rag_result.get('answer', 'No answer generated')}")

        # Print metadata if available
        if 'sources' in rag_result and rag_result['sources']:
            print(f"\n🔗 Sources used: {len(rag_result['sources'])}")

        print("\n✅ Pipeline test completed successfully!")

    except Exception as e:
        print(f"❌ Error during pipeline test: {str(e)}")
        import traceback
        traceback.print_exc()




async def test_basic_retrieval():
    """
    Test just the retrieval component to see if embeddings exist.
    """
    print("\n" + "="*50)
    print("🔍 Testing Basic Retrieval Only...")
    print("="*50)

    try:
        retrieval_service = RetrievalService()
        await retrieval_service.initialize()

        # Test with a simple query
        test_queries = ["ROS 2", "robotics", "AI"]

        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            sources = await retrieval_service.retrieve_by_query(query=query, top_k=3)

            if sources:
                print(f"  Found {len(sources)} relevant sources")
                for i, source in enumerate(sources[:2], 1):  # Show first 2
                    print(f"    {i}. Score: {source.score:.3f}, Text: {source.text[:80]}...")
            else:
                print(f"  ❌ No sources found for query: {query}")

    except Exception as e:
        print(f"❌ Error in basic retrieval test: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """
    Main function to run all tests.
    """
    print("🚀 RAG Pipeline Diagnostic Tool")
    print(f"📖 Using collection: {settings.qdrant_settings.collection_name}")
    print(f"🔢 Vector size: {settings.qdrant_settings.vector_size}")
    print(f"🌐 Qdrant host: {settings.qdrant_settings.host}")
    print()

    # Run the basic retrieval test first
    await test_basic_retrieval()

    # Run the question & answer test
    await test_question_answer()

    # Run the full pipeline test
    await test_rag_pipeline()

    print("\n" + "="*50)
    print("📋 Test Summary:")
    print("- If basic retrieval finds no sources, embeddings weren't stored properly")
    print("- If sources are found but no intelligent response is generated,")
    print("  there may be an issue with the IntelligenceService")
    print("- If both work, the issue might be in the ChatKit integration")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())