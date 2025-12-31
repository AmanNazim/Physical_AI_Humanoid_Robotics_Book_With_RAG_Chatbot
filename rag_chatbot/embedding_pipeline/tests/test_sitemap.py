#!/usr/bin/env python3
"""
Test script for sitemap processing
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the rag_chatbot directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from embedding_pipeline import EmbeddingPipeline, generate_embeddings_from_sitemap


async def test_sitemap_processing():
    """Test the sitemap processing functionality"""
    print("Testing sitemap processing...")

    # Initialize pipeline
    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # Use the specified sitemap URL from the specification
    sitemap_url = "https://amannazim.github.io/Physical_AI_Humanoid_Robotics_Book_With_RAG_Chatbot/sitemap.xml"

    print(f"Processing sitemap from: {sitemap_url}")

    try:
        result = await generate_embeddings_from_sitemap(sitemap_url)
        print(f"Sitemap processing completed with result: {result}")

        # Print final statistics
        stats = pipeline.get_pipeline_stats()
        print(f"Final statistics: {stats}")

        return result
    except Exception as e:
        print(f"Error during sitemap processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Starting sitemap processing test...")
    result = asyncio.run(test_sitemap_processing())

    if result:
        print("Sitemap processing test completed successfully!")
    else:
        print("Sitemap processing test failed!")
        sys.exit(1)