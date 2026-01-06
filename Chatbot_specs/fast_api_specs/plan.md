# Implementation Plan: FastAPI Subsystem for Global RAG Chatbot System

## 1. Purpose of This Plan

This plan breaks down the **specification.md** into actionable components that define how the FastAPI Subsystem will be built. It outlines:

- Development stages and milestones
- Actionable subcomponents with clear responsibilities
- Dependency ordering for proper implementation sequence
- Integration sequencing for subsystem connections
- Interface-level decisions for clean architecture
- Testing phases to ensure quality
- Deployment preparation for production readiness

This serves as the execution roadmap for the implementation phase, translating requirements into an ordered, actionable blueprint while maintaining strict adherence to the constitutional boundaries.

## 2. High-Level Architecture Plan

The FastAPI Subsystem will follow a structured architecture with clear separation of concerns:

### FastAPI App Instance Structure
- Main application instance with proper lifespan management
- Centralized configuration loading
- Middleware registration in proper order
- Router inclusion with prefix and tags

### Project Directory Layout
```
backend/
├── main.py                 # Application factory
├── app.py                  # App instance
├── config.py               # Settings model
├── middleware/             # Custom middleware
│   ├── __init__.py
│   ├── cors.py             # CORS setup
│   ├── logging.py          # Logging middleware
│   └── rate_limit.py       # Rate limiting
├── routers/                # API routers
│   ├── __init__.py
│   ├── health.py           # Health and config endpoints
│   ├── chat.py             # Chat and streaming endpoints
│   ├── retrieve.py         # Retrieval endpoints
│   └── embed.py            # Embedding ingestion endpoints
├── services/               # Service layer
│   ├── __init__.py
│   ├── retrieval_service.py # Retrieval orchestration
│   ├── rag_service.py      # RAG pipeline orchestration
│   ├── embedding_service.py # Embedding workflow coordination
│   └── streaming_service.py # Streaming response handling
├── schemas/                # Pydantic models
│   ├── __init__.py
│   ├── chat.py             # Chat schemas
│   ├── embedding.py        # Embedding schemas
│   ├── retrieval.py        # Retrieval schemas
│   └── error.py            # Error schemas
└── utils/                  # Utility functions
    ├── __init__.py
    └── logger.py           # Logging setup
```

### Subsystem Abstraction Layers
- Service layer: Orchestrates business logic and subsystem interactions
- API layer: Handles HTTP requests/responses and validation
- Model layer: Defines data structures and validation schemas

### Routers per Functional Domain
- **Health Router**: System health and configuration endpoints
- **Chat Router**: RAG query processing and streaming
- **Retrieve Router**: Pure retrieval operations
- **Embed Router**: Document ingestion triggers

### Service-Layer Classes
- `RetrievalService`: Coordinates vector similarity searches via Database subsystem
- `RAGService`: Manages RAG pipeline orchestration (future-ready for Agents SDK)
- `EmbeddingService`: Coordinates document ingestion with Embeddings subsystem
- `StreamingService`: Handles streaming responses and WebSocket support

### Model & Schema Structure (Pydantic)
- Request models with validation rules
- Response models for consistent output
- Error models for standardized error handling
- Shared base models for common fields

### Subsystem Integration Setup
- DatabaseManager initialization for Qdrant and Postgres access
- EmbeddingProcessor initialization for query embedding generation
- Proper lifecycle management for resource cleanup

### Middleware and Logging Setup
- CORS middleware for ChatKit integration
- JSON logging middleware with trace IDs
- Rate limiting to prevent abuse
- Global exception handling

## 3. Environment & Configuration Plan

### Required Environment Variables
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

### Configuration Structure Plan
- `.env` file loading with python-dotenv
- Settings model using Pydantic BaseSettings
- Environment-specific overrides (dev, staging, prod)
- Secure handling of sensitive information
- Validation of required environment variables

## 4. API Endpoint Development Plan

### 4.1 General Build Steps for Each Endpoint
- Create dedicated router module
- Define request/response Pydantic schemas
- Implement validation rules and constraints
- Call appropriate service layer methods
- Handle errors with standardized responses
- Return properly structured responses
- Ensure type safety throughout

### 4.2 POST /api/v1/embed Endpoint Plan
**Data Flow:**
- Validate request body using Pydantic schema
- Call EmbeddingService to trigger ingestion
- Service calls EmbeddingPipeline for processing
- Return ingestion status with metrics

**Service Invocation Order:**
1. EmbeddingService.validate_document()
2. EmbeddingService.trigger_ingestion()

**Error Capture Pathway:**
- Invalid request → Validation error
- Processing failure → Service error
- Subsystem failure → Integration error
- All errors → Standardized error model

**Logging Points:**
- Request received
- Document validation complete
- Ingestion triggered
- Response sent

### 4.3 POST /api/v1/retrieve Endpoint Plan
**Data Flow:**
- Validate query parameters
- Call RetrievalService to orchestrate retrieval
- Service generates query embedding using Embeddings subsystem
- Service calls Database subsystem for similarity search
- Return retrieved sources with metadata

**Service Invocation Order:**
1. RetrievalService.validate_query()
2. EmbeddingProcessor.generate_embeddings() for query
3. DatabaseManager.query_embeddings()
4. RetrievalService.format_response()

**Error Capture Pathway:**
- Invalid query → Validation error
- Search failure → Database error
- Embedding generation failure → Embeddings error

**Logging Points:**
- Query received
- Embedding generation started
- Search executed
- Response sent

### 4.4 POST /api/v1/chat Endpoint Plan
**Data Flow:**
- Validate query parameters
- Call RAGService to orchestrate RAG flow
- Service uses RetrievalService for context retrieval
- Service prepares context for future Intelligence subsystem
- Return answer with sources

**Service Invocation Order:**
1. RAGService.validate_query_and_context()
2. RetrievalService.retrieve_by_query()
3. RAGService._generate_answer_with_context() (placeholder for future Agents SDK)

**Error Capture Pathway:**
- Invalid query → Validation error
- Retrieval failure → Retrieval error
- Processing failure → Service error

**Logging Points:**
- Query received
- Context retrieval started
- Answer generation started
- Response sent

### 4.5 POST /api/v1/chat/stream Endpoint Plan
**Data Flow:**
- Validate query parameters
- Call RAGService to orchestrate RAG flow
- Use StreamingService for token streaming
- Stream sources and tokens as they become available
- Send completion status

**Service Invocation Order:**
1. RetrievalService.retrieve_by_query()
2. StreamingService.stream_response()
3. Stream response chunks via Server-Sent Events

**Performance Constraints:**
- Support proper streaming with client disconnect handling
- Include heartbeat messages for connection maintenance

### 4.6 GET /api/v1/health Endpoint Plan
**Data Flow:**
- Check Database subsystem connectivity
- Return health status with service information

**Service Invocation Order:**
1. Check DatabaseManager connectivity
2. Format health response

### 4.7 GET /api/v1/config Endpoint Plan
**Data Flow:**
- Return safe frontend configuration
- Include feature flags and UI hints

**Service Invocation Order:**
1. Format configuration response

### 4.8 WebSocket: /api/v1/chat/ws/{session_id} Endpoint Plan
**Data Flow:**
- Establish WebSocket connection
- Receive query from client
- Process query with streaming response
- Send token chunks as they arrive
- Send final answer and sources

**Implementation Plan:**
- Use FastAPI WebSocket support
- Implement async streaming
- Handle connection lifecycle
- Manage session state with StreamingService

## 5. Subsystem Integration Plan

### 5.1 Database Subsystem Integration Plan
- **Integration Point**: Use DatabaseManager singleton from rag_chatbot.databases.database_manager
- **Initialization**: Call connect_all() during service initialization
- **Search Operations**: Use database_manager.query_embeddings() for similarity search
- **Metadata Operations**: Use database_manager.get_chunks_by_document() for metadata retrieval
- **Connection Management**: Respect Database subsystem's lifecycle management

### 5.2 Embeddings Subsystem Integration Plan
- **Integration Point**: Use EmbeddingProcessor from rag_chatbot.embedding_pipeline.gemini_client
- **Initialization**: Call initialize() during service initialization
- **Query Embedding**: Use generate_embeddings() for query vector generation
- **Ingestion**: Use EmbeddingPipeline.process_content() for document processing
- **Error Handling**: Implement proper error propagation from embeddings subsystem

## 6. Request/Response Schema Plan

### Schema Creation Process
- Define base models with common fields
- Create specific request models for each endpoint
- Implement response models with proper typing
- Establish error model hierarchy

### Schema Validation Strategy
- Use Pydantic for automatic validation
- Implement custom validators for complex rules
- Ensure type safety throughout the application
- Validate both requests and responses

### Shared Base Models
- Common fields across multiple schemas
- Reusable validation rules
- Consistent error handling patterns

### Naming Conventions
- Request models: `{EndpointName}Request`
- Response models: `{EndpointName}Response`
- Error models: `{ErrorType}Error`

## 7. Middleware & Cross-Cutting Concerns Plan

### CORS Setup for ChatKit
- Configure allowed origins from environment settings
- Support credentials if needed
- Set appropriate headers for security

### JSON Logging Middleware
- Add structured logging with request/response data
- Include trace IDs for distributed tracing
- Support different log levels

### Rate Limiting Middleware
- Implement per-IP rate limiting
- Configure from environment settings
- Return appropriate error responses

### Error Handling Middleware
- Catch unhandled exceptions
- Return standardized error responses
- Log errors with appropriate context

### Global Exception Handlers
- Handle specific exception types
- Return appropriate HTTP status codes
- Maintain consistent error format

## 8. Authentication & Security Plan

### API Key Authentication Plan
- Create dependency for API key validation (optional)
- Validate keys against environment configuration
- Apply to protected endpoints if needed
- Return appropriate error for invalid keys

### Security Headers
- Implement security middleware
- Add HTTPS enforcement in production
- Set appropriate security headers
- Sanitize input data

### Rate Limiting
- Implement per-minute request limits per IP
- Configure from environment settings
- Prevent abuse of the API

## 9. Performance & Scalability Plan

### Async-First Architecture
- Use async/await throughout the application
- Implement non-blocking I/O operations
- Leverage FastAPI's async capabilities

### Connection Management
- Reuse connections where possible
- Respect subsystem connection management
- Minimize connection overhead

### Performance Goals
- **Health**: < 0.1 seconds
- **Retrieve**: < 1.0 seconds
- **Chat**: < 2.0 seconds (response time depends on LLM call)
- **Streaming**: Proper token-by-token delivery

### Optimization Strategies
- Use efficient data serialization
- Minimize data processing overhead
- Optimize subsystem call patterns

## 10. Streaming & WebSocket Plan

### Server-Sent Events Implementation
- Implement proper SSE format for ChatKit UI
- Include proper event and data formatting
- Support client disconnect handling
- Include heartbeat messages

### WebSocket Implementation
- Support bidirectional communication
- Handle connection lifecycle properly
- Support session management
- Include proper error handling

## 11. Logging & Observability Plan

### Structured Logging (JSON)
- Implement JSON logging format
- Include relevant request/response data
- Add trace IDs for correlation

### Log Categories
- **Request/Response**: Log all API interactions
- **Errors**: Capture all exceptions with context
- **Performance**: Track response times
- **Subsystem Integration**: Log calls to other subsystems

### Metrics Collection
- Response time metrics
- Error rate tracking
- Request volume monitoring
- Subsystem health metrics

## 12. Testing Plan

### Unit Tests
- **Retrieval**: Test retrieval logic and validation
- **RAG**: Test RAG pipeline orchestration
- **Embedding**: Test ingestion coordination
- **Streaming**: Test streaming response handling
- **Schema Validation**: Test Pydantic models

### Integration Tests
- **Database Integration**: Test Database subsystem calls
- **Embeddings Integration**: Test Embeddings subsystem calls
- **API Contracts**: Validate request/response shapes

### End-to-End Tests
- **RAG Pipeline**: Test complete query flow
- **Document Ingestion**: Test complete ingestion flow
- **Streaming**: Test streaming endpoints
- **WebSocket**: Test WebSocket endpoints

## 13. Deployment Plan

### Dockerfile Creation Plan
- Multi-stage build for optimization
- Proper dependency management
- Security scanning integration
- Environment configuration

### Uvicorn Configuration
- Production-ready ASGI server setup
- Performance tuning parameters
- Health check endpoints

### Platform Deployment Options
- **Railway**: Container-based deployment
- **Render**: Web service deployment
- **Fly.io**: Global edge deployment
- **VPS**: Self-hosted deployment

### Secrets Management
- Environment variable handling
- Secure credential storage
- Configuration management

## 14. Future Integration Preparation

### Agents SDK Readiness
- Prepare clean data structures for Agents SDK consumption
- Maintain compatibility with future Intelligence subsystem
- Implement proper context preparation

### ChatKit UI Compatibility
- Support streaming responses for real-time UI
- Maintain proper error handling for frontend
- Include source attribution for citations

## 15. Acceptance Criteria for Successful Implementation

The FastAPI Subsystem is complete when:

- [ ] All routers created according to functional domains
- [ ] All Pydantic schema models defined and validated
- [ ] All subsystem integrations functional and tested
- [ ] All validation rules enforced at appropriate layers
- [ ] All endpoints operational with proper responses
- [ ] Logging fully implemented with structured format
- [ ] Security measures applied and validated
- [ ] Deployment-ready container produced
- [ ] All unit tests pass with adequate coverage
- [ ] All integration tests pass
- [ ] All streaming endpoints work properly
- [ ] All WebSocket endpoints work properly
- [ ] All interfaces respect constitutional boundaries
- [ ] Performance goals achieved
- [ ] Health checks operational
- [ ] Error handling consistent across all endpoints
- [ ] CORS properly configured for ChatKit integration
- [ ] Rate limiting implemented and functional