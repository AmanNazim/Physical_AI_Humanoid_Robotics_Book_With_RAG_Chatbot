# Specification: FastAPI Subsystem for Global RAG Chatbot System

## 1. Purpose of This Specification

This document translates the FastAPI Subsystem Constitution into actionable, verifiable technical requirements that define the API definitions, data contracts, integration rules, required CRUD operations, validation rules, expected behaviors, and subsystem boundaries. This specification defines what must be built without including implementation details.

## 2. System Overview (High-Level Architecture)

The FastAPI Subsystem serves as the **central orchestration layer** for the RAG Chatbot system. It orchestrates the flow of data between multiple subsystems:

- **ChatKit (frontend)**: Receives user queries and returns responses
- **Database Subsystem**: Coordinates vector similarity searches (Qdrant) and metadata operations (Neon Postgres)
- **Embeddings Subsystem**: Triggers document ingestion and embedding generation workflows
- **Future Intelligence Subsystem (Agents SDK)**: Forwards queries with context for reasoning (integration-ready)

The RAG loop pipeline orchestrated by FastAPI involves: receiving user queries → validating input → requesting vector retrieval from Database subsystem → receiving metadata enrichment → preparing inputs for future Intelligence subsystem → returning final answers with citations.

## 3. API Versioning Requirements

All endpoints must be namespaced under `/api/v1/` to ensure version compatibility. Future versions must support `/api/v2/` without breaking existing `/api/v1/` functionality. Versioning cannot be omitted and must follow RESTful conventions.

## 4. Core Functional Requirements

FastAPI must implement the following core functions:

### 4.1. Document Ingestion Flow
Endpoints must allow:
- Triggering embedding generation through the embeddings subsystem
- Coordinating with Database subsystem for vector and metadata storage
- Returning ingestion status and metrics

### 4.2. RAG Query Pipeline
FastAPI must support endpoints for:
1. Accepting user queries with proper validation
2. Requesting vector retrieval from Database subsystem (Qdrant)
3. Receiving metadata from Database subsystem (Postgres)
4. Preparing context for future Intelligence subsystem
5. Returning answers with citations, context chunks, and latency metrics

## 5. Detailed Endpoint Specifications

All endpoint schemas must be rigorously defined with URL, method, request schema, response schema, validation rules, error cases, and integration responsibilities.

### 5.1. `POST /api/v1/embed`
Trigger embedding ingestion workflow through the embeddings subsystem.

**Request Body:**
- `text` (string, required)
- `document_metadata` (object, optional)

**FastAPI Responsibilities:**
- Validate request parameters
- Call Embeddings subsystem pipeline to process content
- Return ingestion status and metrics

**Response:**
- `status: "success" | "error"`
- `message`
- `chunks_processed`
- `embeddings_generated`

### 5.2. `POST /api/v1/retrieve`
Pure retrieval endpoint (no LLM processing).

**Request Body:**
- `query` (string, required)
- `top_k` (int, default 5)
- `filters` (object, optional)

**Flow:**
1. Validate input parameters
2. Generate query embedding using Embeddings subsystem
3. Call Database subsystem for similarity search
4. Return retrieved sources with metadata

**Response:**
- `sources: [{chunk_id, document_id, text, score, metadata}]`

### 5.3. `POST /api/v1/chat`
Main RAG endpoint and orchestrator.

**Request Body:**
- `query` (string, required)
- `max_context` (int, default 5)
- `session_id` (string, optional)

**Flow:**
1. Validate input parameters
2. Generate query embedding using Embeddings subsystem
3. Call Database subsystem for similarity search
4. Prepare context for future Intelligence subsystem
5. Return result with sources

**Response:**
- `answer`
- `sources: [{chunk_id, document_id, text, score, metadata}]`
- `session_id`
- `query`

### 5.4. `POST /api/v1/chat/stream`
Streaming RAG endpoint with Server-Sent Events.

**Request Body:**
- `query` (string, required)
- `max_context` (int, default 5)
- `session_id` (string, optional)

**Flow:**
1. Validate input parameters
2. Generate query embedding using Embeddings subsystem
3. Call Database subsystem for similarity search
4. Stream tokens as they become available
5. Return sources and completion status

**Response:**
- Server-Sent Events stream with:
  - Source information
  - Token chunks
  - Completion status

### 5.5. `GET /api/v1/health`
Returns system health diagnostics.

**Response Body:**
- `status: "ok"`
- `service: "RAG Chatbot API"`
- `qdrant_connected: boolean`
- `postgres_connected: boolean`
- `version: "1.0.0"`

### 5.6. `GET /api/v1/config`
Returns safe frontend configuration.

**Response Body:**
- `feature_flags: {}`
- `streaming_enabled: boolean`
- `ui_hints: {}`

### 5.7. WebSocket Endpoint: `/api/v1/chat/ws/{session_id}`
For real-time streaming responses.

**Requirements:**
- Async WebSocket connection
- Stream tokens as they arrive
- Handle bidirectional communication
- Must send:
  - Connection confirmation
  - Source information
  - Token chunks during processing
  - Final answer
  - List of sources
  - Completion status

## 6. Data Validation Requirements

FastAPI must use Pydantic models for all request and response schemas:
- Request schemas for all incoming data
- Response schemas for all outgoing data
- Database interaction models for communication with the database subsystem
- Error models for standardized error responses

**Validation Rules:**
- Reject requests with empty text content
- Reject queries with empty content
- Enforce correct data types for all fields
- Respond with standardized error models

## 7. Error Handling Requirements

Define a **uniform error model** for all endpoints:

```
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

All error responses must follow this structure with appropriate codes and messages.

## 8. Integration with Subsystems

### 8.1. Database Subsystem Integration Rules
FastAPI must use the DatabaseManager abstraction:
- `database_manager.query_embeddings(query_vector, top_k, filters)`
- `database_manager.get_chunks_by_document(document_id)`
- `database_manager.connect_all()`
- `database_manager.close_all()`

FastAPI must NOT:
- Bypass the Database subsystem's abstraction layer
- Make direct database calls
- Handle connection management directly (delegate to Database subsystem)

### 8.2. Embeddings Subsystem Integration Rules
FastAPI must:
- Call `EmbeddingPipeline.process_content(text, document_reference)` for ingestion
- Use `EmbeddingProcessor.generate_embeddings(chunks)` for query embedding generation
- Respect the embeddings subsystem's processing contracts

Embeddings subsystem provides:
- Chunking and preprocessing
- Embedding generation
- Vector storage coordination

### 8.3. Future Intelligence Layer Integration
FastAPI prepares structured data for:
- `query` (user's question)
- `retrieved_context_chunks` (from Database subsystem)
- `session_id` (if provided)

Prepared for future consumption by Intelligence subsystem:
```
{
  "query": "...",
  "context_chunks": [...],
  "session_id": "..."
}
```

FastAPI currently implements placeholder logic but must be ready for Intelligence subsystem integration.

## 9. Authentication & Security Requirements

FastAPI must support:
- API key-based authentication for external services
- Environment-based secrets management
- Rejection of unauthorized requests
- Sanitization of all user input to prevent injection attacks
- Enforcement of HTTPS in production environments
- Proper CORS configuration for ChatKit frontend integration
- Rate limiting to prevent abuse

## 10. Logging Specifications

FastAPI must log:
- HTTP method and URL for each request
- Request processing start time
- Response processing end time
- Errors and exceptions with appropriate context
- Unique trace ID per request for distributed tracing
- Integration call results with other subsystems

Logs must be:
- JSON structured for easy parsing
- Production-safe with appropriate log levels
- Privacy-safe without exposing sensitive user data

## 11. Performance Specifications

FastAPI must:
- Use async (`async def`) endpoints to handle concurrent requests
- Reuse connection pools for database and external service connections
- Minimize synchronous blocking operations
- Support up to ~200 requests per second on mid-tier servers
- Respond within specified time limits:
  - Health API: < 0.1 seconds
  - Retrieve API: < 1.0 seconds
  - Chat API: < 2.0 seconds (response time depends on LLM call)

Streaming responses must support proper token-by-token delivery.

## 12. Deployment Requirements

FastAPI must support:
- Docker containerization for consistent deployment
- Uvicorn + Gunicorn for production-grade ASGI server capabilities
- Deployable on multiple platforms:
  - Railway
  - Fly.io
  - Render
  - HuggingFace Spaces (inference endpoints)
  - VPS (Ubuntu)
- `.env` file configuration loading
- Horizontal scaling readiness for increased load
- Proper lifecycle management with startup/shutdown events

## 13. Streaming & WebSocket Requirements

FastAPI must implement:
- Server-Sent Events (SSE) for streaming responses
- WebSocket support for real-time communication
- Proper token streaming for ChatKit UI compatibility
- Client disconnect handling
- Heartbeat messages for connection maintenance
- Proper stream closure

## 14. Testing Requirements

FastAPI must include:
- Unit tests for each endpoint with various input scenarios
- Integration tests with Database and Embeddings subsystems using mocks
- Schema validation tests to ensure API contract compliance
- Contract tests for ChatKit frontend integration
- Streaming endpoint tests
- Load test plan (optional but recommended)

## 15. Forbidden Behaviors (From Constitution)

FastAPI must NOT:
- Compute embeddings directly (delegate to embeddings subsystem)
- Perform LLM inference or reasoning (delegate to intelligence subsystem)
- Bypass database subsystem for direct data access
- Manipulate vectors directly (delegate to database subsystem)
- Contain business logic that belongs to other subsystems
- Modify chunking behavior (delegate to embeddings subsystem)
- Expose internal server errors directly to clients
- Return undocumented responses that don't match defined schemas
- Perform direct vector searches (use Database subsystem)

## 16. Future Integration Requirements

FastAPI must be designed to support:
- Clean integration with Agents SDK (proper data structures, context preparation)
- ChatKit UI streaming compatibility (proper SSE format, WebSocket support)
- Scalable architecture for high-concurrency scenarios

## 17. Acceptance Criteria

The FastAPI subsystem is complete when:
- All endpoints exist as defined in Section 5
- All request and response schemas are validated and functional
- All integration rules with Database and Embeddings subsystems are respected
- Error responses follow the standardized error model
- Logs are structured according to specifications
- All subsystem connections are established and functional
- Streaming endpoints work with ChatKit UI compatibility
- WebSocket endpoints support real-time communication
- A fully deployable container exists with proper configuration
- All tests pass with appropriate coverage