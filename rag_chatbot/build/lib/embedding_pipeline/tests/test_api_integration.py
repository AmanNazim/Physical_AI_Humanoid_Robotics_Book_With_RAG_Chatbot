"""
Minimal integration test to verify the Gemini embedding API is working.
This test checks that the API can generate embeddings for sample text.
"""
import asyncio
import os
from pathlib import Path

# Add the parent directory to the path so we can import from the pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gemini_client import get_gemini_client


def test_api_connection():
    """Test basic API connection and embedding generation."""
    print("Testing Gemini API connection...")

    # Check if API key is set
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  GEMINI_API_KEY environment variable not set")
        print("Please set GEMINI_API_KEY to run online tests")
        return False

    print("✓ GEMINI_API_KEY found")
    return True


async def test_single_embedding():
    """Test generating a single embedding."""
    print("\nTesting single embedding generation...")

    try:
        client = get_gemini_client()

        # Test text
        test_text = "This is a test sentence for embedding generation."

        print(f"Input text: '{test_text}'")

        # Generate single embedding
        embedding = await client.embed_single(test_text)

        print(f"Generated embedding with {len(embedding)} dimensions")

        # Verify embedding properties
        assert len(embedding) > 0, "Embedding should not be empty"
        assert all(isinstance(val, float) for val in embedding), "All values should be floats"

        print("✓ Single embedding test passed")
        return True

    except Exception as e:
        print(f"✗ Single embedding test failed: {str(e)}")
        return False


async def test_batch_embedding():
    """Test generating embeddings for a batch of texts."""
    print("\nTesting batch embedding generation...")

    try:
        client = get_gemini_client()

        # Test texts
        test_texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is the third test sentence."
        ]

        print(f"Input texts: {test_texts}")

        # Generate batch embeddings
        embeddings = await client.embed_batch(test_texts)

        print(f"Generated {len(embeddings)} embeddings")

        # Verify batch properties
        assert len(embeddings) == len(test_texts), "Should have same number of embeddings as inputs"
        assert all(len(embedding) > 0 for embedding in embeddings), "All embeddings should not be empty"
        assert all(all(isinstance(val, float) for val in embedding) for embedding in embeddings), "All values should be floats"

        print("✓ Batch embedding test passed")
        return True

    except Exception as e:
        print(f"✗ Batch embedding test failed: {str(e)}")
        return False


async def run_online_test():
    """Run the minimal online test to check API functionality."""
    print("="*60)
    print("GEMINI EMBEDDING API INTEGRATION TEST")
    print("="*60)

    # Test API connection
    if not test_api_connection():
        print("\nSkipping online tests due to missing API key")
        print("Set GEMINI_API_KEY environment variable to run API tests")
        return False

    # Run embedding tests
    single_success = await test_single_embedding()
    batch_success = await test_batch_embedding()

    print("\n" + "="*60)
    print("TEST RESULTS:")
    print(f"Single embedding: {'✓ PASS' if single_success else '✗ FAIL'}")
    print(f"Batch embedding:  {'✓ PASS' if batch_success else '✗ FAIL'}")

    overall_success = single_success and batch_success
    print(f"Overall result:   {'✓ PASS' if overall_success else '✗ FAIL'}")
    print("="*60)

    return overall_success


if __name__ == "__main__":
    success = asyncio.run(run_online_test())
    if success:
        print("\n✓ API integration test completed successfully!")
    else:
        print("\n✗ API integration test had failures.")
        exit(1 if not success else 0)