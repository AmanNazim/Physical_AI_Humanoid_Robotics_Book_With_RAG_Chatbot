"""
Debug script to identify exactly where the hanging occurs in the process_and_store function
"""
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_process_and_store():
    """Debug version of process_and_store to identify hanging points"""
    logger.info("Debugging process_and_store function...")

    # Sample content and metadata
    content = "This is a test document for debugging purposes. " * 100  # Create test content
    metadata = {
        "document_reference": "Test Document",
        "page_reference": "test://debug",
        "section_title": "Test Section",
        "processing_version": "1.0",
        "additional_metadata": {
            "source_url": "test://debug",
            "content_type": "test",
            "language": "en",
            "module": "debug",
            "batch_group": "test"
        }
    }

    logger.info("Starting process_and_store debugging...")

    # Step 1: Import required modules
    logger.info("Step 1: Importing required modules...")
    from .chunker import clean_text
    from .utils import generate_content_hash
    logger.info("✓ Modules imported successfully")

    # Step 2: Normalize text
    logger.info("Step 2: Normalizing text...")
    start_time = asyncio.get_event_loop().time()
    try:
        normalized_text = clean_text(content)
        logger.info(f"✓ Text normalized in {asyncio.get_event_loop().time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"✗ Text normalization failed: {e}")
        return False

    # Step 3: Chunk text
    logger.info("Step 3: Chunking text...")
    start_time = asyncio.get_event_loop().time()
    try:
        from .chunker import chunk_text
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, chunk_text, normalized_text)
        logger.info(f"✓ Text chunked in {asyncio.get_event_loop().time() - start_time:.2f}s, {len(chunks)} chunks generated")
    except Exception as e:
        logger.error(f"✗ Text chunking failed: {e}")
        return False

    if not chunks:
        logger.error("✗ No valid chunks generated")
        return False

    # Step 4: Generate embeddings
    logger.info("Step 4: Generating embeddings...")
    start_time = asyncio.get_event_loop().time()
    try:
        from .embedder import batch_generate
        # Add timeout protection
        embeddings = await asyncio.wait_for(batch_generate(chunks), timeout=120.0)
        logger.info(f"✓ Embeddings generated in {asyncio.get_event_loop().time() - start_time:.2f}s, {len(embeddings)} embeddings")
    except asyncio.TimeoutError:
        logger.error(f"✗ Embedding generation timed out after {asyncio.get_event_loop().time() - start_time:.2f}s")
        return False
    except Exception as e:
        logger.error(f"✗ Embedding generation failed: {e}")
        return False

    # Verify embeddings
    if len(embeddings) != len(chunks):
        logger.error(f"✗ Mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks")
        return False

    # Step 5: Initialize Qdrant
    logger.info("Step 5: Initializing Qdrant...")
    start_time = asyncio.get_event_loop().time()
    qdrant_client = None
    try:
        from .config import config
        from .pipeline import initialize_qdrant
        qdrant_client = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, initialize_qdrant),
            timeout=30.0
        )
        logger.info(f"✓ Qdrant initialized in {asyncio.get_event_loop().time() - start_time:.2f}s")
    except asyncio.TimeoutError:
        logger.error(f"✗ Qdrant initialization timed out after {asyncio.get_event_loop().time() - start_time:.2f}s")
        if qdrant_client:
            try:
                qdrant_client.close()
            except:
                pass
        return False
    except Exception as e:
        logger.error(f"✗ Qdrant initialization failed: {e}")
        if qdrant_client:
            try:
                qdrant_client.close()
            except:
                pass
        return False

    # Step 6: Create collection if not exists
    logger.info("Step 6: Creating collection if not exists...")
    start_time = asyncio.get_event_loop().time()
    try:
        from .pipeline import create_collection_if_not_exists
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, create_collection_if_not_exists, qdrant_client),
            timeout=60.0
        )
        logger.info(f"✓ Collection check completed in {asyncio.get_event_loop().time() - start_time:.2f}s")
    except asyncio.TimeoutError:
        logger.error(f"✗ Collection creation timed out after {asyncio.get_event_loop().time() - start_time:.2f}s")
        if qdrant_client:
            try:
                qdrant_client.close()
            except:
                pass
        return False
    except Exception as e:
        logger.error(f"✗ Collection creation failed: {e}")
        if qdrant_client:
            try:
                qdrant_client.close()
            except:
                pass
        return False

    # Step 7: Prepare and store embeddings in Qdrant
    logger.info("Step 7: Preparing and storing embeddings in Qdrant...")
    start_time = asyncio.get_event_loop().time()
    try:
        import uuid
        from .pipeline import store_embedding_in_qdrant

        # Prepare points for Qdrant
        payloads = []
        chunk_ids = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)

            # Generate content hash for integrity
            content_hash = generate_content_hash(chunk)

            # Create payload with metadata
            payload = {
                "id": chunk_id,
                "chunk": chunk,
                "document_reference": metadata.get("document_reference", ""),
                "page_reference": metadata.get("page_reference", None),
                "section_title": metadata.get("section_title", ""),
                "processing_version": metadata.get("processing_version", "1.0"),
                "content_hash": content_hash,
                **metadata.get("additional_metadata", {})
            }

            payloads.append(payload)

        logger.info(f"Prepared {len(chunk_ids)} embeddings for storage")

        # Store embeddings in Qdrant using the dedicated function with timeout
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, store_embedding_in_qdrant, embeddings, payloads, chunk_ids),
            timeout=300.0  # 5 minutes for Qdrant upload
        )
        logger.info(f"✓ Qdrant storage completed in {asyncio.get_event_loop().time() - start_time:.2f}s")

    except asyncio.TimeoutError:
        logger.error(f"✗ Qdrant storage timed out after {asyncio.get_event_loop().time() - start_time:.2f}s")
        if qdrant_client:
            try:
                qdrant_client.close()
            except:
                pass
        return False
    except Exception as e:
        logger.error(f"✗ Qdrant storage failed: {e}")
        if qdrant_client:
            try:
                qdrant_client.close()
            except:
                pass
        return False
    finally:
        if qdrant_client:
            try:
                qdrant_client.close()
            except:
                pass

    logger.info("🎉 All steps completed successfully! No hanging detected in this flow.")
    return True

async def main():
    success = await debug_process_and_store()
    if success:
        logger.info("🎉 Debug test completed successfully - no hanging points found in the tested flow!")
    else:
        logger.info("❌ Debug test identified hanging points.")

if __name__ == "__main__":
    asyncio.run(main())