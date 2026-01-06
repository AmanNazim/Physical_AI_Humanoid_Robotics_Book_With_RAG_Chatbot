# RAG Chatbot FastAPI Backend

This is a production-ready FastAPI backend that serves as the central orchestration layer for the RAG (Retrieval-Augmented Generation) system. It provides REST endpoints for the ChatKit UI, integrates with the Database subsystem for retrieval operations, and coordinates with the Embeddings subsystem for ingestion workflows.

## Architecture

The backend follows a clean architecture with clear separation of concerns:

- **API Layer**: Handles HTTP requests/responses and validation
- **Service Layer**: Orchestrates business logic and subsystem interactions
- **Model Layer**: Defines data structures and validation schemas
- **Middleware Layer**: Handles cross-cutting concerns (CORS, logging, rate limiting)

## Core Components

### Service Layer
- `RetrievalService`: Coordinates vector similarity searches via Database subsystem
- `RAGService`: Manages RAG pipeline orchestration (future-ready for Agents SDK)
- `EmbeddingService`: Coordinates document ingestion with Embeddings subsystem
- `StreamingService`: Handles streaming responses and WebSocket support

### API Endpoints
- `/api/v1/health`: Health check and system diagnostics
- `/api/v1/config`: Safe frontend configuration
- `/api/v1/embed`: Trigger embedding ingestion workflow
- `/api/v1/retrieve`: Pure retrieval endpoint (no LLM processing)
- `/api/v1/chat`: Main RAG endpoint and orchestrator
- `/api/v1/chat/stream`: Streaming RAG endpoint with Server-Sent Events
- `/api/v1/chat/ws/{session_id}`: WebSocket endpoint for real-time communication

## Integration with Subsystems

### Database Subsystem
- Uses `DatabaseManager` from `rag_chatbot.databases.database_manager`
- Coordinates with both Qdrant (for vector search) and Neon Postgres (for metadata) through the Database subsystem
- Respects the Database subsystem's query and storage contracts

### Embeddings Subsystem
- Uses `EmbeddingPipeline` and `EmbeddingProcessor` from `rag_chatbot.embedding_pipeline`
- Triggers embedding generation through the EmbeddingPipeline interface
- Respects the Embeddings subsystem's processing contracts

## Configuration

The application uses Pydantic BaseSettings for configuration management with the following environment variables:

- `QDRANT_URL`: URL for Qdrant vector database
- `QDRANT_API_KEY`: Authentication key for Qdrant
- `QDRANT_COLLECTION_NAME`: Collection name for embeddings
- `QDRANT_VECTOR_SIZE`: Size of embedding vectors
- `NEON_POSTGRES_URL`: Connection string for Neon Postgres
- `GEMINI_API_KEY`: API key for Gemini embeddings
- `GEMINI_MODEL`: Gemini embedding model name
- `EMBEDDING_DIMENSION`: Output dimensionality for embeddings
- `LLM_API_KEY`: API key for LLM (future use)
- `LLM_MODEL`: LLM model to use (future use)
- `LLM_BASE_URL`: LLM API base URL (future use)
- `FASTAPI_SECRET_KEY`: Secret key for security
- `API_KEY`: API key for authentication (optional)
- `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins
- `HOST`: Host for the server
- `PORT`: Port for the server
- `RELOAD`: Enable auto-reload during development
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `RATE_LIMIT_REQUESTS_PER_MINUTE`: Number of requests allowed per minute per IP

## Running the Application

```bash
cd rag_chatbot/backend
python main.py
```

The application will start on `http://localhost:8000` by default.

## Future Integration

The backend is designed to support:
- Clean integration with Agents SDK (proper data structures, context preparation)
- ChatKit UI streaming compatibility (proper SSE format, WebSocket support)
- Scalable architecture for high-concurrency scenarios