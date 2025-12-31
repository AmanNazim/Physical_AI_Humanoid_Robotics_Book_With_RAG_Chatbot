# Specification: Intelligence Layer (OpenAI Agents SDK) Subsystem for Global RAG Chatbot System

## 1. Subsystem Purpose

The Intelligence Layer Subsystem serves as the cognitive reasoning engine for the entire RAG system. This subsystem is responsible for receiving structured queries from FastAPI, applying retrieval-augmented reasoning using pre-fetched context from Qdrant and Neon, and generating grounded responses using the Agents SDK and free OpenRouter LLM key. The subsystem streams incremental token-level responses to the frontend (ChatKit) while enforcing prompt and context engineering best practices. It maintains optional session memory for multi-turn conversations while respecting the constitutional boundaries that prohibit direct database access or embedding generation.

## 2. Input/Output Specifications

### Inputs
1. `user_query` — string, the user's question that requires reasoning and response generation
2. `context_chunks` — array of pre-retrieved text chunks from Qdrant and Neon databases, containing relevant information for the query
3. `session_id` — optional string for maintaining session-based memory across multi-turn conversations
4. `persona_config` — optional JSON defining tone, verbosity, and style parameters for response generation

### Outputs
1. `response_stream` — incremental token-level stream of the generated response for real-time delivery
2. `final_response` — fully assembled response object with:
   - `text`: string containing the complete response text
   - `sources`: array of chunk references that were used to ground the response
   - `structured_data`: optional JSON schema if specifically requested
3. `metadata` — including latency metrics, token usage statistics, and warnings if guardrails were triggered during generation

## 3. Interfaces & Subsystem Connections

### FastAPI Interface
- Receives structured query requests from the FastAPI backend
- Returns responses through FastAPI to the frontend (ChatKit)
- Handles session management and conversation state as provided by FastAPI

### Embeddings Subsystem Interface
- Receives pre-embedded context chunks from the Embeddings subsystem via FastAPI
- Does NOT perform embedding generation itself - only consumes pre-processed context
- Validates the format and quality of received embeddings

### Database Subsystem Interface (Qdrant & Neon)
- Uses only context provided by upstream systems, does not query databases directly
- Does NOT perform direct vector searches or metadata retrieval
- Relies entirely on context passed from FastAPI

### Agents SDK Interface
- Core orchestration of reasoning and LLM calls using the OpenAI Agents SDK
- Manages agent state and conversation flow
- Handles tool usage and planning capabilities when needed

### OpenRouter LLM Key Management
- Secure storage of API key in environment variables or secure vault
- Automatic rotation and retry mechanisms for free tier rate limits
- Token usage monitoring and quota management
- Implementation of backoff strategies to prevent rate limiting

## 4. Prompt Engineering Specifications

### Modular Prompt Structure
The subsystem implements a modular prompt structure with three distinct components:
1. **Instruction template**: Defines persona, task requirements, and behavioral guidelines
2. **Context template**: Incorporates retrieved chunks and optional session memory
3. **Query template**: Formats the user question and specifies output requirements

### Guardrail Implementation
- Validates output length to prevent excessively long responses
- Ensures proper citation of source chunks in all responses
- Prevents hallucinations by enforcing strict grounding in provided context
- Applies appropriate tone and style based on persona configuration
- Supports dynamic persona overrides per individual request

## 5. Context Engineering Specifications

### Context Processing
- Truncates or summarizes long context to fit within token limits
- Implements chunk prioritization to place most relevant information first
- Applies overlap control to avoid context loss during processing
- Provides optional session memory injection for multi-turn dialogue management
- Formats context to ensure proper interpretation by the agent system

### Context Quality Assurance
- Validates the relevance and quality of provided context chunks
- Ensures proper formatting and structure for agent consumption
- Implements deduplication to avoid redundant information
- Maintains context coherence across multiple chunks

## 6. Guardrails

### Domain Compliance
- Ensures all outputs respect domain knowledge (Physical AI, ROS2, Simulation, VLA)
- Enforces safe, instructional tone in all responses
- Implements retry or reformulation logic for invalid outputs
- Prevents generation of content outside the specified domain expertise

### Safety and Quality Controls
- Validates response accuracy against provided context
- Ensures proper source attribution for all claims
- Implements content filtering to prevent inappropriate outputs
- Maintains factual accuracy within domain expertise

## 7. Streaming & Performance

### Streaming Requirements
- Responses must be streamed to the client token-by-token for real-time interaction
- Implements async handling to ensure low-latency responses (<1.5s per query ideally)
- Performs internal chunked response assembly before final formatting
- Provides optional caching for repeated queries to improve efficiency

### Performance Optimization
- Implements efficient prompt construction to minimize token usage
- Optimizes API call patterns to reduce latency
- Provides connection pooling for LLM services
- Implements intelligent retry mechanisms for failed requests

## 8. API Key Management (OpenRouter)

### Security and Storage
- Securely stores and manages free-tier API keys using environment variables or secure vault
- Implements automatic key rotation and security protocols
- Enforces rate limiting to prevent key bans and service disruption
- Logs usage metrics including token counts and API call frequency

### Reliability and Scaling
- Implements retry logic with exponential backoff for failed API calls
- Provides monitoring and alerting for quota usage
- Allows future integration with multiple keys for load balancing if quotas are exceeded
- Implements circuit breaker patterns to handle API service disruptions

## 9. Performance & Latency Targets

### Response Time Requirements
- Average query processing latency: <1.5 seconds for streamed responses
- Full response assembly: <2.5 seconds for complete responses
- Multi-turn conversation memory load: <2% additional latency per turn
- Streaming throughput: >50 tokens per second for real-time interaction

### Resource Utilization
- Maintains efficient token usage to optimize API costs
- Implements caching strategies to reduce redundant processing
- Optimizes memory usage for session state management
- Provides monitoring for resource consumption and performance metrics

## 10. Acceptance Criteria

### Functional Requirements
- Subsystem correctly receives structured requests from FastAPI with proper validation
- Applies persona and prompt engineering specifications as defined
- Streams token-level responses to ChatKit with minimal latency
- Correctly integrates OpenRouter free-tier LLM key management with security protocols
- Implements guardrails to prevent hallucinations and enforce context fidelity
- Logs all interactions for monitoring, debugging, and compliance purposes
- Maintains specified latency targets and performance metrics consistently

### Quality and Reliability
- All responses are properly grounded in provided context with appropriate citations
- Session management works correctly for multi-turn conversations
- Error handling is robust with appropriate fallback strategies
- Security measures prevent API key exposure and enforce proper access controls
- Performance remains stable under expected load conditions