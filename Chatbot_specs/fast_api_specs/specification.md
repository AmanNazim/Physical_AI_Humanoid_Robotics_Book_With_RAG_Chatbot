# Specification: FastAPI Subsystem for Global RAG Chatbot System

## 1. Purpose of This Specification

This document translates the FastAPI Subsystem Constitution into actionable, verifiable technical requirements that define the API definitions, data contracts, integration rules, required CRUD operations, validation rules, expected behaviors, and subsystem boundaries. This specification defines what must be built without including implementation details.

## 2. System Overview (High-Level Architecture)

The FastAPI Subsystem serves as the **central backend entrypoint** for the RAG Chatbot system. It orchestrates the flow of data between multiple subsystems:

- **ChatKit (frontend)**: Receives user queries and returns responses
- **Intelligence Subsystem (OpenAI Agents SDK)**: Forwards queries with context for reasoning
- **Embeddings Subsystem (Cohere)**: Triggers embedding generation for new documents
- **Qdrant Vector DB Subsystem**: Requests vector similarity searches
- **Neon Postgres DB Subsystem**: Retrieves metadata for context and stores document information

The RAG loop pipeline orchestrated by FastAPI involves: receiving user queries → validating input → requesting vector retrieval from Qdrant → retrieving metadata from Postgres → forwarding context + query to Intelligence subsystem → returning final answer with citations.

## 3. API Versioning Requirements

All endpoints must be namespaced under `/api/v1/` to ensure version compatibility. Future versions must support `/api/v2/` without breaking existing `/api/v1/` functionality. Versioning cannot be omitted and must follow RESTful conventions.

## 4. Core Functional Requirements

FastAPI must implement the following core functions:

### 4.1. Document Ingestion Flow
Endpoints must allow:
- Uploading raw text content
- Uploading markdown documents
- Uploading PDFs (optional)
- Saving document metadata to Postgres
- Triggering embedding generation through the embeddings subsystem
- Storing resulting vectors in Qdrant via the database subsystem
- Returning ingestion status and metrics

### 4.2. RAG Query Pipeline
FastAPI must support endpoints for:
1. Accepting user queries with proper validation
2. Requesting vector retrieval from Qdrant subsystem
3. Receiving document metadata from Postgres subsystem
4. Forwarding context and query to Intelligence subsystem
5. Returning final answers with citations, context chunks, and latency metrics

## 5. Detailed Endpoint Specifications

All endpoint schemas must be rigorously defined with URL, method, request schema, response schema, validation rules, error cases, and integration responsibilities.

### 5.1. `POST /api/v1/ingest/text`
Upload raw text to the system.

**Request Body:**
- `document_id` (string, optional → auto-generate if missing)
- `title` (string)
- `source` (string: "manual" | "pdf" | "md")
- `text` (string)

**FastAPI Responsibilities:**
- Validate request parameters
- Save metadata to Postgres via DB subsystem
- Call Embeddings subsystem to generate embeddings
- Send vectors to Qdrant via database subsystem
- Return ingestion summary

**Response:**
- `status: "success"`
- `document_id`
- `chunks_created`
- `vectors_stored`
- `elapsed_ms`

### 5.2. `POST /api/v1/query`
User queries the chatbot.

**Request Body:**
- `query` (string, required)
- `max_context` (int, default 5)
- `session_id` (string, optional)

**Flow:**
1. Validate input parameters
2. Call Qdrant subsystem for similarity search
3. Retrieve metadata from Postgres
4. Forward query + context to Intelligence subsystem
5. Return result with sources

**Response:**
- `answer`
- `sources: [{chunk_id, document_id, text}]`
- `latency_ms`

### 5.3. `GET /api/v1/documents`
Returns all documents stored in the system.

**Response Body:**
Array of document metadata:
- `document_id`
- `title`
- `source`
- `chunk_count`

### 5.4. `GET /api/v1/health`
Returns simple health diagnostics.

**Response Body:**
- `status: "ok"`
- `qdrant_connected: boolean`
- `postgres_connected: boolean`
- `version: "v1"`

### 5.5. Optional WebSocket Endpoint: `/api/v1/ws/chat`
For streaming agent responses.

**Requirements:**
- Async WebSocket connection
- Stream tokens as they arrive
- Must send:
  - Token chunks during processing
  - Final answer
  - List of sources

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

### 8.1. Qdrant Integration Rules
FastAPI must use the Qdrant subsystem abstraction:
- `qdrant.insert_vectors(chunks)`
- `qdrant.search(query_embedding, top_k)`

FastAPI must NOT:
- Compute embeddings directly
- Handle vector storage directly

### 8.2. Neon Postgres Integration Rules
FastAPI must use the DB abstraction layer:
- `db.save_document()`
- `db.save_chunk()`
- `db.get_document_metadata()`

FastAPI must NOT:
- Write raw SQL queries
- Manage database transactions manually

### 8.3. Embeddings Subsystem Integration Rules
FastAPI must:
- Call `embeddings.generate(text)` when document ingestion is requested

Embeddings subsystem returns:
```
{
  "chunks": [...],
  "vectors": [...]
}
```

### 8.4. Intelligence Layer Integration
FastAPI forwards:
- `query`
- `retrieved_context_chunks`
- `session_id` (if provided)

And expects:
```
{
  "answer": "...",
  "source_ids": [...]
}
```

FastAPI must not perform reasoning or LLM calls directly.

## 9. Authentication & Security Requirements

FastAPI must support:
- API key-based authentication for external services
- Environment-based secrets management
- Rejection of unauthorized requests
- Sanitization of all user input to prevent injection attacks
- Enforcement of HTTPS in production environments
- Proper CORS configuration for ChatKit frontend integration

## 10. Logging Specifications

FastAPI must log:
- HTTP method and URL for each request
- Request processing start time
- Response processing end time
- Errors and exceptions with appropriate context
- Unique trace ID per request for distributed tracing

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
  - Ingestion API: < 2.5 seconds
  - Query API: < 1.5 seconds

Caching is optional but recommended for repeated vector lookups.

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

## 13. Testing Requirements

FastAPI must include:
- Unit tests for each endpoint with various input scenarios
- Integration tests with Qdrant and Postgres using mocks
- Schema validation tests to ensure API contract compliance
- Contract tests for ChatKit frontend integration
- Load test plan (optional but recommended)

## 14. Forbidden Behaviors (From Constitution)

FastAPI must NOT:
- Compute embeddings directly (delegate to embeddings subsystem)
- Perform LLM inference or reasoning (delegate to intelligence subsystem)
- Bypass database subsystem for direct data access
- Manipulate vectors directly (delegate to database subsystem)
- Contain business logic that belongs to other subsystems
- Modify chunking behavior (delegate to embeddings subsystem)
- Expose internal server errors directly to clients
- Return undocumented responses that don't match defined schemas

## 15. Acceptance Criteria

The FastAPI subsystem is complete when:
- All endpoints exist as defined in Section 5
- All request and response schemas are validated and functional
- All integration rules with other subsystems are respected
- Error responses follow the standardized error model
- Logs are structured according to specifications
- All subsystem connections are established and functional
- A fully deployable container exists with proper configuration
- All tests pass with appropriate coverage