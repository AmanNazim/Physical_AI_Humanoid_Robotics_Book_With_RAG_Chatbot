# Constitution: Intelligence Layer (OpenAI Agents SDK) Subsystem for Global RAG Chatbot System

## 1. Subsystem Mission

The Intelligence Layer Subsystem serves as the **cognitive reasoning engine** for the entire RAG system. This subsystem is responsible for processing user queries forwarded by the FastAPI backend, performing sophisticated reasoning over retrieved context chunks from Qdrant and Neon Postgres, and generating accurate, coherent, and contextually-aware responses. The Intelligence Layer acts as the critical decision-making component that transforms raw retrieved information into meaningful, grounded responses while maintaining conversational context and applying appropriate safety measures.

The mission of the Intelligence Layer is to serve as the reasoning and response generation hub that connects user queries to relevant knowledge, ensuring that all outputs are grounded in the retrieved context and aligned with the system's constitutional requirement for non-hallucinated, fact-based responses. The subsystem maintains the constitutional requirement that all responses must be traceable to source documents while applying appropriate prompt engineering and safety guardrails.

## 2. Core Responsibilities

The Intelligence Layer must:

**Query Processing:**
- Process user queries received from the FastAPI backend
- Apply reasoning algorithms over provided context chunks
- Generate contextually-aware and accurate responses
- Maintain session-based state for ongoing conversations
- Stream responses incrementally to the frontend via FastAPI or WebSockets

**Prompt Engineering:**
- Construct optimized prompts using best practices in prompt engineering
- Apply dynamic persona definition for the agent (tone, verbosity, domain specialization)
- Implement context engineering for intelligent context selection and relevance scoring
- Apply safety and validation instructions to prevent hallucinations

**Response Generation:**
- Generate grounded, non-hallucinated responses based on retrieved context
- Provide structured answers with proper citations to source chunks
- Apply chain-of-thought reasoning when complex analysis is required
- Implement fallback strategies for handling ambiguous queries

## 3. Strict Subsystem Boundaries

The Intelligence Layer must NOT:

- Handle embedding generation - this belongs to the Embeddings subsystem
- Directly access Postgres or Qdrant databases - must receive context from FastAPI
- Perform vector searches or similarity matching - this belongs to the Database subsystem
- Store or manage document metadata - this belongs to the Database subsystem
- Process raw documents or text content for embedding - this belongs to the Embeddings subsystem
- Bypass FastAPI for direct communication with other subsystems

The Intelligence Layer ONLY performs reasoning, summarization, instruction following, and answer generation using pre-retrieved context. It maintains strict separation of concerns by consuming context from other subsystems without accessing their underlying storage or processing mechanisms.

## 4. API Surface Governance

The Intelligence Layer must:

**Interface Management:**
- Expose only documented interfaces for receiving query-context pairs from FastAPI
- Accept structured query requests with pre-retrieved context chunks
- Return structured responses with citations and confidence indicators
- Maintain stable API contracts that maintain backward compatibility

**Response Standards:**
- Require consistent response formatting following established patterns
- Include proper citations to source context chunks in all responses
- Never expose internal reasoning processes or agent state in responses
- Support streaming response format for real-time user feedback

## 5. Integration Rules with Other Subsystems

### Intelligence Layer → FastAPI Subsystem
- Must receive structured query-context payloads from FastAPI
- Must return responses through FastAPI to the frontend
- Must respect session management provided by FastAPI
- Must handle error conditions and timeouts from upstream

### Intelligence Layer → Database Subsystem (Qdrant + Neon)
- Must NOT directly access database systems
- Must receive all context through FastAPI backend
- Must NOT bypass established retrieval pathways
- Must validate context integrity received from upstream systems

### Intelligence Layer → Embeddings Subsystem
- Must NOT directly request embedding generation
- Must consume pre-processed context from upstream systems
- Must NOT trigger document processing or chunking operations
- Must respect the embedding format and metadata provided by upstream

## 6. Security Requirements

The Intelligence Layer must:

**API Key Management:**
- Use OpenRouter API key instead of default OpenAI API key for LLM services
- Securely store and manage API credentials in environment variables
- Implement proper API key rotation and security protocols
- Never expose API keys in logs or responses

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
- Implement optional caching for repeated context usage
- Optimize prompt length to reduce API costs and latency
- Support concurrent user sessions without performance degradation

## 8. Reliability & Stability

The Intelligence Layer must:

**Error Handling:**
- Handle all LLM API failures with appropriate fallback strategies
- Return structured error formats that are consistent across the system
- Maintain session continuity during partial processing failures
- Implement retry logic for transient API failures

**Compatibility:**
- Guarantee backward compatibility for existing response formats
- Ensure deterministic response generation for identical inputs
- Implement graceful degradation when context quality is poor
- Maintain consistent persona and tone across sessions

## 9. Observability Rules

The Intelligence Layer must include:

**Logging and Monitoring:**
- Implement structured logging for all query-response interactions
- Track response quality metrics and hallucination detection
- Monitor processing latency and token generation rates
- Include trace IDs for distributed tracing across subsystems

**Quality Assurance:**
- Log citation accuracy and source chunk relevance
- Track user satisfaction indicators where available
- Monitor for potential bias or inappropriate content generation
- Maintain audit logs for compliance and debugging

## 10. Deployment Requirements

The Intelligence Layer must support:

**Infrastructure:**
- Containerized deployment using standard container technologies
- Running behind a production-grade orchestration system
- Compatibility with serverless platforms when needed
- Environment-based configuration for different deployment targets

**Scalability:**
- Support horizontal scaling for increased query load
- Implement connection pooling for LLM API services
- Optimize API usage to manage rate limits effectively
- Maintain consistent behavior across different deployment targets

## 11. Forbidden Actions

The Intelligence Layer MUST NOT:

- Generate embeddings or trigger embedding operations
- Directly access or query database systems (Qdrant, Neon)
- Perform document processing, chunking, or text extraction
- Store or persist any user data or conversation history
- Bypass the FastAPI subsystem for direct frontend communication
- Generate content not grounded in provided context chunks
- Modify or transform retrieved context before processing
- Access external knowledge sources beyond provided context
- Generate responses that cannot be traced to source documents

The Intelligence Layer is a reasoning and response generation layer ONLY, with no data storage, retrieval, or preprocessing capabilities.

## 12. Non-Negotiable Architectural Principles

The Intelligence Layer must operate under:

**Design Principles:**
- Stateless processing per query (except optional session memory)
- Single-responsibility principle - only handle reasoning and response generation
- Strict contract-first interface design with FastAPI
- No circular dependencies with other subsystems

**Safety and Quality:**
- Complete grounding requirement - all responses must be based on provided context
- Hallucination prevention through strict source citation requirements
- Safety-first approach with comprehensive guardrail implementation
- Quality assurance through relevance and accuracy validation

## 13. Final Constitutional Guarantee

This Constitution represents the **unchangeable governing rules** for the Intelligence Layer (OpenAI Agents SDK) Subsystem. All future Specifications, Plans, Tasks, and Implementation generated by Claude Code MUST strictly follow this Constitution. No deviations are allowed. This document establishes the fundamental architectural boundaries, responsibilities, and constraints that govern the Intelligence Layer's role within the Global RAG Chatbot System. Any implementation that violates these principles is considered non-compliant with the system architecture and must be corrected to maintain system integrity.