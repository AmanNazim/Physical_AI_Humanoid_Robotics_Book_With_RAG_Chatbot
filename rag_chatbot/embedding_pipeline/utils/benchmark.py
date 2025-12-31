#!/usr/bin/env python3
"""
Benchmark script for embeddings pipeline
"""
import asyncio
import time
import sys
from pathlib import Path

# Add the rag_chatbot directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from embedding_pipeline import EmbeddingPipeline


async def benchmark_chunking():
    """Benchmark chunking performance"""
    print("Benchmarking chunking performance...")

    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # Create sample text of various sizes
    sample_sizes = [1000, 5000, 10000, 20000]  # characters

    for size in sample_sizes:
        sample_text = "This is a sample text for benchmarking. " * (size // 50)

        start_time = time.time()
        chunks = pipeline.chunk_processor.chunk_text(sample_text, f"benchmark_doc_{size}")
        end_time = time.time()

        processing_time = end_time - start_time
        chunks_per_second = len(chunks) / processing_time if processing_time > 0 else 0

        print(f"Size: {size} chars -> {len(chunks)} chunks in {processing_time:.2f}s ({chunks_per_second:.2f} chunks/s)")


async def benchmark_embedding_generation():
    """Benchmark embedding generation performance"""
    print("\nBenchmarking embedding generation performance...")

    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    # Create sample chunks of various sizes
    sample_texts = [
        "This is a short sample text for embedding.",
        "This is a medium-length sample text for embedding. " * 10,
        "This is a longer sample text for embedding. " * 50,
    ]

    for i, text in enumerate(sample_texts):
        # Create a chunk manually
        from embedding_pipeline.base_classes import Chunk
        import hashlib

        chunk_id = hashlib.sha256(f"{text}{i}".encode()).hexdigest()[:16]
        chunk = Chunk(
            chunk_id=chunk_id,
            content=text,
            token_count=pipeline.chunk_processor.engine.estimate_token_count(text),
            character_start=0,
            character_end=len(text),
            token_start=0,
            token_end=pipeline.chunk_processor.engine.estimate_token_count(text),
            document_reference=f"benchmark_chunk_{i}"
        )

        start_time = time.time()
        embeddings = await pipeline.embedding_processor.generate_embeddings([chunk])
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"Chunk {i+1} ({len(text)} chars) -> embedding in {processing_time:.2f}s")


async def run_benchmarks():
    """Run all benchmarks"""
    print("Starting embeddings pipeline benchmarks...")

    await benchmark_chunking()
    await benchmark_embedding_generation()

    print("\nBenchmarks completed!")


if __name__ == "__main__":
    asyncio.run(run_benchmarks())