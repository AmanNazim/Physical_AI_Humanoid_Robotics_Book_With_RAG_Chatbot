# FastAPI Backend Subsystem for Global RAG Chatbot System

This is the FastAPI Backend Subsystem for the Global RAG Chatbot System. It serves as the central orchestration layer for the RAG system, providing REST endpoints for the ChatKit UI, OpenAI Agents runtime, vector retrieval, metadata queries, and pipeline orchestration.

## Features

- **API Gateway**: Central backend gateway for the entire RAG system
- **RAG Pipeline**: Orchestrates the flow of data between multiple subsystems
- **Async Processing**: Built with async/await for high concurrency
- **Streaming Support**: Server-Sent Events (SSE) for real-time responses
- **Security**: API key authentication and rate limiting
- **Observability**: Structured logging with request IDs and timing
- **Health Checks**: Comprehensive health and readiness endpoints

## API Endpoints

### Health & Configuration
- `GET /api/v1/health` - Health check
- `GET /api/v1/ready` - Readiness check
- `GET /api/v1/status` - Detailed system status
- `GET /api/v1/config` - System configuration for frontend

### Document Ingestion
- `POST /api/v1/embed-text` - Ingest raw text content
- `POST /api/v1/add-document` - Add document with chunk → embed → store workflow

### Search & Retrieval
- `POST /api/v1/search` - Vector search wrapper
- `POST /api/v1/semantic-search` - Semantic search
- `POST /api/v1/hybrid-search` - Hybrid search (Neon + Qdrant)

### Chat & Query
- `POST /api/v1/chat` - Main RAG endpoint for chat functionality
- `POST /api/v1/conversation-state` - Conversation state management

### Documents
- `GET /api/v1/documents` - List all documents
- `GET /api/v1/document/{id}` - Get specific document
- `DELETE /api/v1/document/{id}` - Delete document

### WebSocket
- `WS /api/v1/ws/chat` - Streaming chat responses

## Environment Variables

Create a `.env` file with the following variables:

```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=book_embeddings
QDRANT_VECTOR_SIZE=1536

# Neon PostgreSQL Configuration
NEON_POSTGRES_URL=postgresql://user:password@localhost:5432/dbname

# Google Gemini Embedding Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=1536

# LLM Configuration
LLM_API_KEY=your_llm_api_key_here
LLM_MODEL=openai/gpt-4-turbo
LLM_BASE_URL=https://openrouter.ai/api/v1

# FastAPI Security Configuration
FASTAPI_SECRET_KEY=your-super-secret-key-here
API_KEY=your-api-key-here

# CORS Configuration
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Service Configuration
SERVICE_NAME=fastapi-backend
LOG_LEVEL=INFO

# RAG Configuration
MAX_CONTEXT_CHUNKS=5
RETRIEVAL_TOP_K=5
RETRIEVAL_THRESHOLD=0.7
```

## Installation & Setup

1. Clone the repository
2. Navigate to the fastapi_backend directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables (see above)
5. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t rag-chatbot-fastapi-backend .

# Run the container
docker run -p 8000:8000 --env-file .env rag-chatbot-fastapi-backend
```

## Testing

Run the tests:

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# All tests
pytest
```

## Architecture

The backend follows a clean architecture pattern with the following layers:

- **API Layer**: FastAPI endpoints with request/response validation
- **Service Layer**: Business logic and workflow orchestration
- **Client Layer**: Subsystem integrations (Qdrant, Postgres, Embeddings, Intelligence)
- **Model Layer**: Pydantic models for data validation
- **Middleware Layer**: CORS, logging, error handling, authentication

## Integration with Other Subsystems

- **Qdrant Vector DB**: For vector similarity searches
- **Neon Postgres**: For metadata storage and retrieval
- **Embeddings Subsystem**: For generating document embeddings
- **Intelligence Subsystem**: For LLM-based reasoning and response generation

## Security

- API key authentication for protected endpoints
- Rate limiting to prevent abuse
- Input validation using Pydantic models
- Secure handling of sensitive information

## Performance

- Async-first architecture for high concurrency
- Connection pooling for database and external service connections
- Minimal overhead in request handling
- Efficient resource usage for concurrent users