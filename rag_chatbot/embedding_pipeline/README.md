# Embeddings & Chunking Pipeline

This module implements the embeddings generation and storage pipeline for the RAG Chatbot system. It handles document ingestion, text preprocessing, chunking, embedding generation using Google Gemini API, and storage to Qdrant vector database.

## Architecture

The pipeline follows a modular, class-based architecture:

- **Config**: Configuration management and validation
- **Base Classes**: Core data structures and abstract base classes
- **URL Crawler**: Sitemap parsing and URL crawling functionality
- **File Processor**: Document ingestion from various file formats
- **Text Preprocessor**: Text normalization and sanitization
- **Chunking Engine**: Dynamic chunking with 800-1200 token range and overlap
- **Gemini Client**: Google Gemini API integration for embeddings
- **Database**: Qdrant and Neon database integration
- **Pipeline**: Main orchestration class

## Features

- **Document Ingestion**: Supports crawling from sitemap URLs and file processing
- **Text Preprocessing**: Unicode normalization, HTML tag removal, content sanitization
- **Dynamic Chunking**: Splits text into 800-1200 token chunks with 200-token overlap
- **Embedding Generation**: Uses Google Gemini API for high-quality embeddings
- **Database Storage**: Stores embeddings to Qdrant vector database and metadata to Neon PostgreSQL
- **Asynchronous Processing**: Efficient async/await patterns throughout
- **Error Handling**: Comprehensive error handling with retry logic
- **Monitoring**: Processing statistics and logging

## Usage

### Command Line Interface

```bash
# Process a single document file
python -m embedding_pipeline.main path/to/document.pdf

# Process from a sitemap URL
python -m embedding_pipeline.main --sitemap https://example.com/sitemap.xml

# Process a single URL
python -m embedding_pipeline.main --url https://example.com/page.html
```

### Programmatic Usage

```python
from embedding_pipeline import EmbeddingPipeline, generate_embeddings_for_document, generate_embeddings_from_sitemap

# Initialize and process
pipeline = EmbeddingPipeline()
await pipeline.initialize()

# Process content directly
result = await pipeline.process_content("Your text content here", document_reference="my_doc")

# Process from file or URL
result = await generate_embeddings_for_document("path/to/document.txt")

# Process from sitemap
result = await generate_embeddings_from_sitemap("https://example.com/sitemap.xml")
```

## Configuration

The pipeline uses environment variables for configuration:

```env
GEMINI_API_KEY=your_google_gemini_api_key
EMBED_MODEL_NAME=gemini-embedding-001
CHUNK_SIZE_MIN=800
CHUNK_SIZE_MAX=1200
CHUNK_OVERLAP=200
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=postgresql://user:password@host:port/database
BATCH_SIZE=5
```

## Processing Flow

1. **Document Ingestion**: Load content from files or URLs
2. **Text Preprocessing**: Normalize and sanitize text
3. **Chunking**: Split text into 800-1200 token chunks with overlap
4. **Embedding Generation**: Generate embeddings using Google Gemini API
5. **Storage**: Store embeddings to Qdrant and metadata to Neon
6. **Statistics**: Track processing metrics

## Error Handling

The pipeline includes comprehensive error handling:
- Retry logic with exponential backoff for API calls
- Content validation and sanitization
- Graceful degradation when components fail
- Detailed logging for debugging

## Performance Considerations

- Batching API calls to optimize Gemini API usage
- Asynchronous processing throughout
- Efficient database operations
- Memory management for large documents