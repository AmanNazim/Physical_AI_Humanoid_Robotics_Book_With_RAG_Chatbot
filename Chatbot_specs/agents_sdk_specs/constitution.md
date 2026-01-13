# Constitution: Intelligence Layer (OpenAI Agents SDK) Subsystem for Global RAG Chatbot System

## 1. Subsystem Mission

The Intelligence Layer Subsystem serves as the **cognitive reasoning engine** for the entire RAG system. This subsystem is responsible for processing user queries forwarded by the FastAPI backend, performing sophisticated reasoning over retrieved context chunks from Qdrant and PostgreSQL, and generating accurate, coherent, and contextually-aware responses. The Intelligence Layer acts as the critical decision-making component that transforms raw retrieved information into meaningful, grounded responses while maintaining conversational context and applying appropriate safety measures.

The mission of the Intelligence Layer is to serve as the reasoning and response generation hub that connects user queries to relevant knowledge, ensuring that all outputs are grounded in the retrieved context and aligned with the system's constitutional requirement for non-hallucinated, fact-based responses. The subsystem maintains the constitutional requirement that all responses must be traceable to source documents while applying appropriate prompt engineering and safety guardrails.

**Integration with Existing Subsystems**: The Intelligence Layer must integrate seamlessly with the existing RAG architecture where:
- FastAPI backend handles API routing and request validation
- RetrievalService handles document retrieval from Qdrant and PostgreSQL
- EmbeddingPipeline handles vector generation and processing
- DatabaseManager handles unified access to Qdrant and PostgreSQL
- RAGService orchestrates the retrieval and generation flow

## 2. Core Responsibilities

The Intelligence Layer must:

**Query Processing:**
- Process user queries received from the FastAPI backend via RAGService
- Apply reasoning algorithms over context chunks provided by RetrievalService
- Generate contextually-aware and accurate responses using OpenAI Agents SDK
- Maintain session-based state for ongoing conversations
- Stream responses incrementally to the frontend via FastAPI or WebSockets
- Integrate with existing streaming_service for response streaming

**Prompt Engineering:**
- Construct optimized prompts using best practices in prompt engineering
- Apply dynamic persona definition for the agent (tone, verbosity, domain specialization)
- Implement context engineering for intelligent context selection and relevance scoring
- Apply safety and validation instructions to prevent hallucinations
- Leverage OpenAI Agents SDK tools and handoff capabilities for complex reasoning

**Response Generation:**
- Generate grounded, non-hallucinated responses based on retrieved context from Qdrant/PostgreSQL
- Provide structured answers with proper citations to source chunks
- Apply chain-of-thought reasoning when complex analysis is required
- Implement fallback strategies for handling ambiguous queries
- Integrate with existing Source schema for response formatting

## 3. Strict Subsystem Boundaries

The Intelligence Layer must NOT:

- Handle embedding generation - this belongs to the EmbeddingPipeline subsystem
- Directly access Postgres or Qdrant databases - must receive context from FastAPI via RetrievalService
- Perform vector searches or similarity matching - this belongs to the DatabaseManager subsystem
- Store or manage document metadata - this belongs to the DatabaseManager subsystem
- Process raw documents or text content for embedding - this belongs to the EmbeddingPipeline subsystem
- Bypass FastAPI for direct communication with other subsystems
- Replace existing RAGService orchestration - must integrate with existing service layer

The Intelligence Layer ONLY performs reasoning, summarization, instruction following, and answer generation using pre-retrieved context from RetrievalService. It maintains strict separation of concerns by consuming context from other subsystems without accessing their underlying storage or processing mechanisms. It must integrate with existing services rather than replacing them.

## 4. API Surface Governance

The Intelligence Layer must:

**Interface Management:**
- Integrate with existing RAGService to receive query-context pairs from FastAPI
- Accept structured query requests with pre-retrieved context chunks from RetrievalService
- Return structured responses with citations and confidence indicators via RAGService
- Maintain stable API contracts that maintain backward compatibility with existing endpoints
- Integrate with existing streaming_service for response streaming capabilities

**Response Standards:**
- Require consistent response formatting following established patterns using Source schema
- Include proper citations to source context chunks in all responses using existing format
- Never expose internal reasoning processes or agent state in responses
- Support streaming response format for real-time user feedback via existing WebSocket/streaming endpoints

## 5. Integration Rules with Other Subsystems

### Intelligence Layer → FastAPI Subsystem
- Must receive structured query-context payloads from FastAPI via RAGService
- Must return responses through RAGService to FastAPI to the frontend
- Must respect session management provided by FastAPI
- Must handle error conditions and timeouts from upstream
- Must integrate with existing WebSocket and streaming endpoints

### Intelligence Layer → Database Subsystem (Qdrant + PostgreSQL)
- Must NOT directly access database systems
- Must receive all context through RetrievalService via RAGService
- Must NOT bypass established retrieval pathways
- Must validate context integrity received from upstream systems

### Intelligence Layer → RAGService
- Must integrate with existing RAGService.generate_response method
- Must accept pre-retrieved context from RetrievalService
- Must return properly formatted responses using Source schema
- Must maintain existing API contract and response structure
- Must handle existing error patterns and fallbacks

### Intelligence Layer → RetrievalService
- Must receive pre-fetched context chunks from RetrievalService via RAGService
- Must NOT perform direct retrieval operations
- Must validate context integrity received from upstream systems
- Must work with existing Source schema for retrieved content

### Intelligence Layer → EmbeddingPipeline Subsystem
- Must NOT directly request embedding generation
- Must consume pre-processed context from RetrievalService
- Must NOT trigger document processing or chunking operations
- Must respect the embedding format and metadata provided by upstream

## 6. Security Requirements

The Intelligence Layer must:

**API Key Management:**
- Use OpenRouter API key instead of default OpenAI API key for LLM services
- Securely store and manage API credentials in environment variables using existing settings
- Implement proper API key rotation and security protocols
- Never expose API keys in logs or responses
- Integrate with existing logging infrastructure for security monitoring

**Content Safety:**
- Apply strict guardrails to prevent hallucinations and fabricated information
- Implement content filtering to avoid sensitive or inappropriate outputs
- Ensure all responses are grounded in provided context chunks
- Implement relevance scoring to maintain response quality

**Input Validation:**
- Validate all incoming query-context pairs for proper format
- Implement rate limiting for processing requests
- Sanitize user queries to prevent injection attacks
- Implement timeout mechanisms for processing operations

**Output Protection:**
- Ensure no sensitive context information is leaked in responses
- Implement proper redaction of potentially sensitive content
- Maintain privacy of user queries and conversation history
- Log all interactions for security monitoring

## 7. Performance Requirements

The Intelligence Layer must guarantee:

**Latency and Efficiency:**
- Provide response streaming with minimal initial delay
- Implement efficient prompt construction and processing
- Optimize token generation for real-time interaction
- Use efficient memory management for session contexts

**Resource Management:**
- Maintain minimal computational overhead during processing
- Integrate with existing caching mechanisms where applicable
- Optimize prompt length to reduce API costs and latency
- Support concurrent user sessions without performance degradation
- Work within existing resource constraints of the RAG system

## 8. Reliability & Stability

The Intelligence Layer must:

**Error Handling:**
- Handle all LLM API failures with appropriate fallback strategies
- Return structured error formats that are consistent with existing system patterns
- Maintain session continuity during partial processing failures
- Implement retry logic for transient API failures
- Integrate with existing error handling patterns in RAGService and FastAPI
- Work with existing exception handling and logging infrastructure

**Compatibility:**
- Guarantee backward compatibility for existing response formats and API contracts
- Ensure deterministic response generation for identical inputs
- Implement graceful degradation when context quality is poor
- Maintain consistent persona and tone across sessions
- Maintain compatibility with existing streaming and WebSocket endpoints
- Preserve existing session management patterns

## 9. Observability Rules

The Intelligence Layer must include:

**Logging and Monitoring:**
- Implement structured logging using existing rag_logger for all query-response interactions
- Track response quality metrics and hallucination detection
- Monitor processing latency and token generation rates
- Include trace IDs for distributed tracing across subsystems
- Integrate with existing logging infrastructure and patterns

**Quality Assurance:**
- Log citation accuracy and source chunk relevance using existing Source schema
- Track user satisfaction indicators where available
- Monitor for potential bias or inappropriate content generation
- Maintain audit logs for compliance and debugging using existing database logging
- Integrate with existing quality assurance patterns and metrics

## 10. Deployment Requirements

The Intelligence Layer must support:

**Infrastructure:**
- Containerized deployment using standard container technologies
- Running within existing FastAPI application structure in rag_chatbot
- Compatibility with serverless platforms when needed
- Environment-based configuration using existing settings module

**Scalability:**
- Support horizontal scaling for increased query load within existing FastAPI structure
- Implement connection pooling for LLM API services
- Optimize API usage to manage rate limits effectively
- Maintain consistent behavior across different deployment targets
- Work within existing resource constraints of the RAG system

## 11. Forbidden Actions

The Intelligence Layer MUST NOT:

- Generate embeddings or trigger embedding operations in EmbeddingPipeline
- Directly access or query database systems (Qdrant, PostgreSQL) - must use RetrievalService
- Perform document processing, chunking, or text extraction
- Store or persist any user data or conversation history directly
- Bypass the RAGService or FastAPI subsystems for direct communication
- Replace existing RAGService orchestration or RetrievalService functionality
- Generate content not grounded in provided context chunks from RetrievalService
- Modify or transform retrieved context before processing
- Access external knowledge sources beyond provided context
- Generate responses that cannot be traced to source documents
- Implement custom session management, must use existing patterns

The Intelligence Layer is a reasoning and response generation layer ONLY, with no data storage, retrieval, or preprocessing capabilities.

## 12. Non-Negotiable Architectural Principles

The Intelligence Layer must operate under:

**Design Principles:**
- Stateless processing per query (except optional session memory through existing patterns)
- Single-responsibility principle - only handle reasoning and response generation
- Strict contract-first interface design with existing RAGService and FastAPI
- No circular dependencies with other subsystems
- Integration-first approach - build upon existing services rather than replacing them

**Safety and Quality:**
- Complete grounding requirement - all responses must be based on provided context from RetrievalService
- Hallucination prevention through strict source citation requirements using existing Source schema
- Safety-first approach with comprehensive guardrail implementation
- Quality assurance through relevance and accuracy validation using existing patterns
- Compliance with existing security and privacy requirements of the RAG system

## 13. Final Constitutional Guarantee

This Constitution represents the **unchangeable governing rules** for the Intelligence Layer (OpenAI Agents SDK) Subsystem. All future Specifications, Plans, Tasks, and Implementation generated by Claude Code MUST strictly follow this Constitution. No deviations are allowed. This document establishes the fundamental architectural boundaries, responsibilities, and constraints that govern the Intelligence Layer's role within the Global RAG Chatbot System. Any implementation that violates these principles is considered non-compliant with the system architecture and must be corrected to maintain system integrity.