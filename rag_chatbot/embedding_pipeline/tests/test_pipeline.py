"""
Test file for embeddings pipeline
"""
import asyncio
import os
from embedding_pipeline import EmbeddingPipeline, generate_embeddings_for_document, generate_embeddings_from_sitemap


async def test_pipeline():
    """Test the embeddings pipeline"""
    print("Testing Embeddings Pipeline...")

    # Create pipeline instance
    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # Test with sample text
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    As machines become increasingly capable, tasks once thought to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect.
    A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
    """

    print("Processing sample text...")
    result = await pipeline.process_content(sample_text, document_reference="test_document")

    print(f"Processing result: {result}")

    # Check pipeline stats
    stats = pipeline.get_pipeline_stats()
    print(f"Pipeline stats: {stats}")


async def test_sitemap_processing():
    """Test sitemap processing - using a sample sitemap URL"""
    print("\nTesting sitemap processing...")

    # Note: Replace with actual sitemap URL when available
    sitemap_url = "https://example.com/sitemap.xml"  # Placeholder - replace with actual URL

    try:
        result = await generate_embeddings_from_sitemap(sitemap_url)
        print(f"Sitemap processing result: {result}")
    except Exception as e:
        print(f"Sitemap processing failed (expected for placeholder URL): {str(e)}")


async def test_file_processing():
    """Test file processing"""
    print("\nTesting file processing...")

    # Create a test file
    test_file_path = "test_document.txt"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write("""
        This is a test document for the embeddings pipeline.
        It contains sample content to test the full processing workflow.
        The pipeline should be able to read this file, process the content,
        chunk it appropriately, generate embeddings, and store them to the database.
        """)

    try:
        result = await generate_embeddings_for_document(test_file_path)
        print(f"File processing result: {result}")
    except Exception as e:
        print(f"File processing error: {str(e)}")
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
    asyncio.run(test_file_processing())
    asyncio.run(test_sitemap_processing())