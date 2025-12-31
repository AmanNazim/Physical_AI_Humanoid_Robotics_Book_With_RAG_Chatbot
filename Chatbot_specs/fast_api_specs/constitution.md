# Constitution: FastAPI Subsystem for Global RAG Chatbot System

## 1. Subsystem Mission

The FastAPI Subsystem serves as the **central backend gateway** for the entire RAG system. This subsystem provides REST endpoints for the ChatKit UI, OpenAI Agents runtime, vector retrieval, metadata queries, and pipeline orchestration. It provides a secure abstraction over Qdrant, Neon Postgres, and Embeddings subsystems, ensuring deterministic, stateless request handling. The FastAPI Subsystem provides a unified interface for all internal RAG services while maintaining strict separation of concerns with other subsystems.

The mission of the FastAPI Subsystem is to act as the orchestration and transport layer that connects all components of the RAG system. It ensures that requests from the frontend are properly validated, routed to the appropriate backend services, and responses are formatted consistently for consumption by the frontend. The subsystem maintains the constitutional requirement for deterministic retrieval by ensuring that all requests follow established pathways without bypassing other subsystems.

## 2. Core Responsibilities

The FastAPI Subsystem must:

**Endpoint Exposure:**
- Expose HTTP endpoints for user query forwarding to Intelligence subsystem
- Provide endpoints for document ingestion triggers
- Enable embedding generation triggers through dedicated endpoints
- Expose RAG retrieval request endpoints
- Manage chat session endpoints
- Provide metadata inspection endpoints

**Validation and Formatting:**
- Enforce API input validation using Pydantic schemas
- Maintain consistent response formatting across all endpoints
- Implement request logging, exception handling, and authentication

**Orchestration:**
- Serve as the **bridge** between Frontend ChatKit UI and backend services
- Coordinate with Database subsystem (Neon) for metadata operations
- Interface with Vector DB subsystem (Qdrant) for retrieval operations
- Integrate with Embeddings subsystem (Cohere) for processing triggers
- Connect with Intelligence Agent subsystem for reasoning operations

## 3. Strict Subsystem Boundaries

The FastAPI Subsystem must NOT:

- Generate embeddings - this belongs to the Embeddings subsystem
- Perform chunking operations - this belongs to the Embeddings subsystem
- Perform vector searches directly (must call Qdrant client layer) - this belongs to the Database subsystem
- Compute LLM responses - this belongs to the Intelligence subsystem
- Contain business logic that belongs to the intelligence layer
- Override subsystem boundaries of other components

The FastAPI Subsystem ONLY orchestrates, exposes API endpoints, validates requests, delegates processing to appropriate subsystems, and returns formatted responses. It maintains strict separation of concerns by delegating all domain-specific operations to specialized subsystems.

## 4. API Surface Governance

The FastAPI Subsystem must:

**Endpoint Management:**
- Expose only documented endpoints as specified in the system contracts
- Produce stable API contracts that maintain backward compatibility
- Version the API using the `/api/v1/...` pattern for clear versioning
- Support WebSocket endpoints for live streaming capabilities

**Response Standards:**
- Require consistent error handling model across all endpoints
- Require structured JSON responses following established patterns
- Never expose internal implementation details in responses
- Support CORS properly for ChatKit UI integration

## 5. Integration Rules with Other Subsystems

### FastAPI → Qdrant Subsystem
- Must use the Qdrant client wrapper provided by the Qdrant subsystem
- Must never embed vectors directly
- Must never compute similarity manually - this is the responsibility of the Qdrant subsystem

### FastAPI → Neon Postgres Subsystem
- Must only call database abstraction functions provided by the Neon subsystem
- Must never write raw SQL unless explicitly defined in DB subsystem specification
- Must never bypass DB validations implemented in the Database subsystem

### FastAPI → Embeddings Subsystem
- Can trigger embedding generation through defined interfaces
- Cannot compute embeddings directly
- Cannot define chunking rules - this is the responsibility of the Embeddings subsystem
- Cannot modify embedding logic - this is the responsibility of the Embeddings subsystem

### FastAPI → Intelligence Layer (Agents)
- Must forward user text queries to the Intelligence subsystem
- Must not create or destroy agent states directly
- Must respect the intelligence subsystem governance and state management

### FastAPI → ChatKit Frontend
- Must expose endpoints for send_message operations
- Must provide retrieve_context endpoints for context retrieval
- Must support generate_response endpoints for response generation
- Must implement show_sources endpoints for citation display
- Must provide load_metadata endpoints for metadata operations

## 6. Security Requirements

The FastAPI Subsystem must:

**Authentication and Authorization:**
- Implement API key authentication or session-based access controls
- Provide rate limiting capability to prevent abuse
- Enforce sanitization of all input to prevent injection attacks

**Data Protection:**
- Implement full request logging for security monitoring
- Ensure no sensitive data exposure in logs or responses
- Secure loading of environment variables containing sensitive information
- Enforce HTTPS in production environments

## 7. Performance Requirements

The FastAPI Subsystem must guarantee:

**Latency and Efficiency:**
- Provide low latency routing for all requests
- Implement async-optimized endpoints for concurrent processing
- Ensure concurrency safety for multiple simultaneous requests
- Use efficient connection pooling for database and external service connections

**Resource Management:**
- Maintain minimal overhead in request handling
- Implement optional caching layer for repeated vector lookups
- Optimize resource usage to support concurrent users

## 8. Reliability & Stability

The FastAPI Subsystem must:

**Error Handling:**
- Handle all exceptions with a unified error handler
- Return structured error formats that are consistent across the system
- Maintain uptime without breaking other subsystems during partial failures

**Compatibility:**
- Guarantee backward compatibility for existing API versions
- Ensure deterministic teardown and startup behavior
- Implement graceful degradation when dependent services are unavailable

## 9. Observability Rules

The FastAPI Subsystem must include:

**Logging and Monitoring:**
- Implement request/response structured logging for debugging and monitoring
- Provide endpoint-level metrics for performance tracking
- Include performance timing for all operations
- Support trace IDs for distributed tracing across subsystems

**Health Management:**
- Expose health-check endpoints for system monitoring
- Provide readiness probe endpoints for deployment orchestration

## 10. Deployment Requirements

The FastAPI Subsystem must support:

**Infrastructure:**
- Containerized deployment using standard container technologies
- Running behind a production-grade ASGI server (uvicorn/gunicorn)
- Compatibility with serverless platforms when needed
- Environment-based configuration switching for different deployment environments

**Portability:**
- Support portability across hosting providers (Railway, Fly.io, Render, etc.)
- Maintain consistent behavior across different deployment targets

## 11. Forbidden Actions

The FastAPI Subsystem MUST NOT:

- Create embeddings - this belongs to the Embeddings subsystem
- Perform agentic reasoning - this belongs to the Intelligence subsystem
- Store files directly - this should go through appropriate storage subsystems
- Bypass vector DB/database subsystems - all data access must go through proper channels
- Contain domain logic belonging to intelligence layer - this belongs to the Intelligence subsystem
- Mutate or transform embeddings - this belongs to the Embeddings subsystem
- Contain UI rendering logic - this belongs to the frontend
- Generate LLM responses - this belongs to the Intelligence subsystem

The FastAPI Subsystem is an orchestration and transport layer ONLY, with no domain-specific processing capabilities.

## 12. Non-Negotiable Architectural Principles

The FastAPI Subsystem must operate under:

**Design Principles:**
- Stateless request model - no session state should be maintained between requests
- Single-responsibility principle - only handle request routing and validation
- Strict contract-first API design - all endpoints must follow established contracts
- No circular dependencies - maintain clear unidirectional data flow

**Security and Isolation:**
- Complete isolation from model weights or embeddings
- SOC2-like logging discipline for security and compliance
- Secure API boundaries that prevent unauthorized access

## 13. Final Constitutional Guarantee

This Constitution represents the **unchangeable governing rules** for the FastAPI Subsystem. All future Specifications, Plans, Tasks, and Implementation generated by Claude Code MUST strictly follow this Constitution. No deviations are allowed. This document establishes the fundamental architectural boundaries, responsibilities, and constraints that govern the FastAPI Subsystem's role within the Global RAG Chatbot System. Any implementation that violates these principles is considered non-compliant with the system architecture and must be corrected to maintain system integrity.