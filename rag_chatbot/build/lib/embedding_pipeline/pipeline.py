"""
Pipeline module for the Embeddings & Chunking Pipeline.
This module orchestrates the complete flow from text input to vector storage.
"""
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import config
from .chunker import chunk_text
from .embedder import batch_generate
from .utils import generate_content_hash
import asyncpg


def initialize_qdrant() -> QdrantClient:
    """
    Initialize Qdrant client with configuration parameters.

    Returns:
        Configured QdrantClient instance
    """
    import time
    start_time = time.time()

    if config.qdrant_api_key:
        client = QdrantClient(
            url=config.qdrant_host,
            api_key=config.qdrant_api_key,
            prefer_grpc=False,  # Disable gRPC for cloud instances to prevent connection issues
            timeout=10.0  # Add timeout to prevent hanging
        )
    else:
        client = QdrantClient(
            host=config.qdrant_host,
            timeout=10.0  # Add timeout to prevent hanging
        )

    # Test connection immediately with timeout by running in executor
    try:
        import concurrent.futures
        import threading

        def test_connection():
            return client.get_collections()

        # Run the connection test in a separate thread with timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(test_connection)
            # Wait for the result with a timeout of 15 seconds
            collections = future.result(timeout=15.0)

        elapsed = time.time() - start_time
        print(f"+ Qdrant client initialized and tested in {elapsed:.2f}s")
    except Exception as e:
        print(f"Qdrant connection test failed: {str(e)}")
        # Still return the client - it will be tested again when used

    return client


def create_collection_if_not_exists(client: QdrantClient):
    """
    Create Qdrant collection if it doesn't exist with proper configuration.

    Args:
        client: QdrantClient instance
    """
    try:
        import time
        import concurrent.futures

        start_time = time.time()

        def get_collections():
            return client.get_collections()

        # Run the collections retrieval in a separate thread with timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(get_collections)
            collections = future.result(timeout=15.0)  # 15 second timeout

        collection_names = [collection.name for collection in collections.collections]
        elapsed = time.time() - start_time
        print(f"Qdrant collections retrieved in {elapsed:.2f}s")

        if config.qdrant_collection_name not in collection_names:
            start_time = time.time()

            def create_collection():
                return client.create_collection(
                    collection_name=config.qdrant_collection_name,
                    vectors_config=models.VectorParams(
                        size=config.gemini_output_dimensionality,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        memmap_threshold=20000,
                        indexing_threshold=20000
                    )
                )

            # Run the collection creation in a separate thread with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(create_collection)
                future.result(timeout=30.0)  # 30 second timeout for collection creation

            elapsed = time.time() - start_time
            print(f"+ Qdrant collection '{config.qdrant_collection_name}' created in {elapsed:.2f}s")
        else:
            print(f"+ Qdrant collection '{config.qdrant_collection_name}' already exists")
    except Exception as e:
        raise RuntimeError(f"Failed to create Qdrant collection: {str(e)}")


def store_embedding_in_qdrant(vectors: List[List[float]], payloads: List[Dict], ids: List[str]):
    """
    Store embeddings in Qdrant with proper error handling and logging.

    Args:
        vectors: List of embedding vectors
        payloads: List of metadata payloads
        ids: List of IDs for the vectors
    """
    client = None
    try:
        # Initialize Qdrant client with timeout
        client = initialize_qdrant()

        # Prepare points
        points = []
        for vector, payload, point_id in zip(vectors, payloads, ids):
            point = models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            points.append(point)

        import time
        import concurrent.futures

        start_time = time.time()

        def perform_upsert():
            return client.upsert(
                collection_name=config.qdrant_collection_name,
                points=points,
                wait=True
            )

        # Run the upsert in a separate thread with timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(perform_upsert)
            # Wait for the result with a timeout of 60 seconds for upsert operation
            future.result(timeout=60.0)

        elapsed = time.time() - start_time
        print(f"+ Qdrant upsert completed in {elapsed:.2f}s for {len(vectors)} vectors")
    except Exception as e:
        raise RuntimeError(f"Failed to store embeddings in Qdrant: {str(e)}")
    finally:
        # Close client connection if it was created
        if client:
            try:
                client.close()
            except:
                pass  # Ignore errors during client close


async def process_and_store(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process text through the complete pipeline: normalize -> chunk -> embed -> store.

    Args:
        text: Input text to process
        metadata: Additional metadata to associate with the text

    Returns:
        Dictionary containing processing results and chunk IDs
    """
    from .chunker import clean_text

    start_time = asyncio.get_event_loop().time()

    # 1. Normalize text
    try:
        normalized_text = clean_text(text)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to normalize text: {str(e)}",
            "chunk_ids": [],
            "processing_time": asyncio.get_event_loop().time() - start_time
        }

    # 2. Chunk text using executor to prevent blocking on tokenization
    try:
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, chunk_text, normalized_text)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to chunk text: {str(e)}",
            "chunk_ids": [],
            "processing_time": asyncio.get_event_loop().time() - start_time
        }

    if not chunks:
        return {
            "status": "error",
            "message": "No valid chunks generated from input text",
            "chunk_ids": [],
            "processing_time": 0
        }

    print(f"   Generated {len(chunks)} chunks for processing")

    # 3. Generate embeddings with timeout protection
    try:
        # Add memory monitoring before processing
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            print(f"   Memory usage high ({memory_percent:.1f}%), proceeding...")
            import gc
            gc.collect()

        # Add timeout to batch_generate operation to prevent hanging
        embeddings = await asyncio.wait_for(
            batch_generate(chunks),
            timeout=300.0  # 5 minutes timeout for batch embedding generation
        )
    except asyncio.TimeoutError:
        return {
            "status": "error",
            "message": "Embedding generation timed out after 5 minutes",
            "chunk_ids": [],
            "processing_time": asyncio.get_event_loop().time() - start_time
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to generate embeddings: {str(e)}",
            "chunk_ids": [],
            "processing_time": asyncio.get_event_loop().time() - start_time
        }

    # Verify that we have embeddings for all chunks
    if len(embeddings) != len(chunks):
        return {
            "status": "error",
            "message": f"Mismatch: generated {len(embeddings)} embeddings for {len(chunks)} chunks",
            "chunk_ids": [],
            "processing_time": asyncio.get_event_loop().time() - start_time
        }

    # 4. Connect to Qdrant and create collection if not exists
    qdrant_client = None
    try:
        # Add timeout to Qdrant initialization - increased timeout to account for internal timeouts
        qdrant_client = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, initialize_qdrant),
            timeout=45.0  # Increased to account for internal 15s timeout
        )

        # Add timeout to collection creation - increased timeout to account for internal timeouts
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, create_collection_if_not_exists, qdrant_client),
            timeout=75.0  # Increased to account for internal 30s timeout
        )
    except asyncio.TimeoutError:
        if qdrant_client:
            try:
                qdrant_client.close()
            except:
                pass
        return {
            "status": "error",
            "message": "Qdrant connection timed out after 45 seconds for init, 75 seconds for collection creation",
            "chunk_ids": [],
            "processing_time": asyncio.get_event_loop().time() - start_time
        }
    except Exception as e:
        if qdrant_client:
            try:
                qdrant_client.close()
            except:
                pass
        return {
            "status": "error",
            "message": f"Failed to connect to Qdrant: {str(e)}",
            "chunk_ids": [],
            "processing_time": asyncio.get_event_loop().time() - start_time
        }

    # 5. Store embeddings + metadata
    try:
        # Prepare points for Qdrant - run in executor to avoid blocking
        def prepare_payloads():
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

                # Periodic progress update for large files
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i+1}/{len(chunks)} chunks...")

            print(f"   Prepared {len(chunk_ids)} embeddings for upload to Qdrant...")
            return payloads, chunk_ids

        # Run payload preparation in executor to avoid blocking
        payloads, chunk_ids = await asyncio.get_event_loop().run_in_executor(None, prepare_payloads)

        # Store embeddings in Qdrant using the dedicated function with timeout
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, store_embedding_in_qdrant, embeddings, payloads, chunk_ids),
                timeout=90.0  # Reduced timeout to account for internal 60s timeout in store_embedding_in_qdrant
            )
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "message": "Qdrant upload timed out after 90 seconds",
                "chunk_ids": [],
                "processing_time": asyncio.get_event_loop().time() - start_time
            }

        # Store metadata in Neon Postgres (using asyncpg directly)
        # Check if database URL is configured before attempting connection
        if config.neon_database_url and "postgresql" in config.neon_database_url:
            print(f"   Storing metadata in Neon Postgres...")
            try:
                conn = await asyncio.wait_for(asyncpg.connect(config.neon_database_url), timeout=30.0)

                # Insert chunk metadata - using only existing columns in the database
                for i, chunk_id in enumerate(chunk_ids):
                    await conn.execute('''
                        INSERT INTO chunks (
                            chunk_id, document_reference, page_reference, section_title,
                            chunk_text, embedding_id, processing_version
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            document_reference = EXCLUDED.document_reference,
                            page_reference = EXCLUDED.page_reference,
                            section_title = EXCLUDED.section_title,
                            chunk_text = EXCLUDED.chunk_text,
                            embedding_id = EXCLUDED.embedding_id,
                            processing_version = EXCLUDED.processing_version
                    ''',
                    chunk_ids[i],
                    metadata.get("document_reference", ""),
                    metadata.get("page_reference"),
                    metadata.get("section_title", ""),
                    chunks[i],
                    chunk_ids[i],  # embedding_id
                    metadata.get("processing_version", "1.0"))

                    # Periodic progress update for large files
                    if (i + 1) % 10 == 0:
                        print(f"   Stored {i+1}/{len(chunk_ids)} metadata entries...")

                await conn.close()
            except asyncio.TimeoutError:
                print("! Database connection or operation timed out, skipping metadata storage")
            except Exception as e:
                print(f"! Failed to store metadata in Neon: {str(e)}, continuing with embeddings only")
        else:
            print("! Neon database URL not configured, skipping metadata storage")

        processing_time = asyncio.get_event_loop().time() - start_time
        print(f"   + Completed processing in {processing_time:.2f}s")

        return {
            "status": "success",
            "message": f"Successfully processed {len(chunks)} chunks",
            "chunk_ids": chunk_ids,
            "processing_time": processing_time
        }

    except Exception as e:
        processing_time = asyncio.get_event_loop().time() - start_time
        return {
            "status": "error",
            "message": f"Failed to store embeddings: {str(e)}",
            "chunk_ids": [],
            "processing_time": processing_time
        }
    finally:
        if qdrant_client:
            try:
                qdrant_client.close()
            except:
                pass


async def process_file(path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a file through the complete pipeline.

    Args:
        path: Path to the file to process
        metadata: Additional metadata to associate with the file

    Returns:
        Dictionary containing processing results and chunk IDs
    """
    try:
        # Read file content with proper encoding detection
        import chardet

        with open(path, 'rb') as file:
            raw_data = file.read()
            encoding_result = chardet.detect(raw_data)
            encoding = encoding_result['encoding'] or 'utf-8'

        with open(path, 'r', encoding=encoding) as file:
            text = file.read()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read file: {str(e)}",
            "chunk_ids": [],
            "processing_time": 0
        }

    # Add file-specific metadata
    file_metadata = {
        **metadata,
        "source_file": path,
        "document_reference": metadata.get("document_reference", path)
    }

    # Process through pipeline
    return await process_and_store(text, file_metadata)