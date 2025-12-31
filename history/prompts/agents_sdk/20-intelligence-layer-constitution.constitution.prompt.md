---
id: 20
title: intelligence-layer-constitution
stage: constitution
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: agents_sdk
branch: main
user: Claude
command: /sp.constitution
labels: [intelligence, agents, constitution, architecture]
links:
  spec: Chatbot_specs/agents_sdk_specs/specification.md
  plan: Chatbot_specs/agents_sdk_specs/plan.md
  ticket: null
  adr: null
  pr: null
files:
 - Chatbot_specs/agents_sdk_specs/constitution.md
tests:
 - null
---

## Prompt

You are **Constitution Writer Expert**, an expert in designing highly structured, formal, and detailed technical constitutions for complex system architectures.

Your task is to generate the **constitution.md** file for the **Intelligence Layer / OpenAI Agents SDK Subsystem** of the RAG Chatbot project.

This document must strictly follow the SDD methodology and define the **subsystem's purpose, boundaries, tools, roles, and operational principles**.
Do not implement anything; only define **what this subsystem is, what it does, and how it integrates**.

Input Reference: (Mandatory To Read):
- Chatbot_specs/constitution.md (important)
- Chatbot_specs/specification.md (important)
- Chatbot_specs/plan.md (important)
- Chatbot_specs/tasks.md (important)
- Chatbot_specs/database_specs/constitution.md
- Chatbot_specs/database_specs/specification.md
- Chatbot_specs/database_specs/plan.md
- Chatbot_specs/database_specs/tasks.md
- Chatbot_specs/embeddings_chunking_specs/constitution.md
- Chatbot_specs/embeddings_chunking_specs/specification.md
- Chatbot_specs/embeddings_chunking_specs/plan.md
- Chatbot_specs/embeddings_chunking_specs/tasks.md
- Chatbot_specs/fast_api_specs/constitution.md
- Chatbot_specs/fast_api_specs/specification.md
- Chatbot_specs/fast_api_specs/plan.md
- Chatbot_specs/fast_api_specs/tasks.md

---

# ============================
# INTELLIGENCE LAYER (OpenAI Agents SDK) — Constitution
# ============================

## **1. Subsystem Purpose**
The Intelligence Layer Subsystem is responsible for:

1. Processing user queries forwarded by the FastAPI backend.
2. Performing reasoning over retrieved context chunks from Qdrant and Neon Postgres.
3. Generating accurate, coherent, and contextually-aware responses.
4. Maintaining session-based state for ongoing conversations.
5. Streaming responses to the frontend via FastAPI or WebSockets for low-latency interaction.
6. Applying prompt engineering, context engineering, and guardrails to ensure safe, relevant, and reliable outputs.

**Boundaries:**
- This subsystem **must not** handle embedding generation.
- This subsystem **must not** directly access Postgres or Qdrant.
- This subsystem **only performs reasoning, summarization, instruction following, and answer generation**.

---

## **2. Subsystem Role**
The Intelligence Layer will:

1. Receive **structured query requests** and context payloads from FastAPI.
2. Construct **optimized prompts** using best practices in **Prompt Engineering** and **Context Engineering**.
3. Apply **dynamic persona definition** for the agent, including tone, verbosity, and domain specialization.
4. Apply **guardrails and safety rules** (filtering, hallucination mitigation, relevance scoring).
5. Stream response tokens incrementally to ensure **real-time user feedback**.
6. Log all interactions for debugging, auditing, and performance analysis.

---

## **3. Tools & Integrations**
The subsystem must integrate the following tools:

### **Mandatory / Best Tools**
1. **OpenAI Agents SDK** — core agent orchestration, planning, and reasoning.
2. **Cohere embeddings** (via FastAPI) — context retrieval for grounding.
3. **Streaming libraries / WebSockets** — for incremental token delivery.
4. **Prompt Engineering frameworks** — for dynamic prompt composition and chaining.
5. **Context Engineering utilities** — for intelligent context selection, truncation, and relevance scoring.

### **Optional / Enhancing Tools**
1. **LangChain** components (if needed) — for chaining tools, memory, and structured reasoning.
2. **RAG orchestration tools** — for advanced retrieval-augmented reasoning.
3. **Cache / performance optimization libraries** — to speed up repeated context usage.

---

## **4. Persona Definition**
The agent persona must be explicitly defined and applied via **Prompt Engineering**:

1. Tone: professional, concise, educational.
2. Domain Knowledge: Physical AI, Robotics, ROS2, Simulation, VLA.
3. Response style: structured, step-by-step, and reference-aware.
4. Guardrails:
   - Avoid hallucinations.
   - Always cite retrieved context chunks.
   - Avoid sensitive content.
5. Dynamic persona adjustments:
   - Adapt to user preference for verbosity.
   - Adapt tone for instructional or conversational scenarios.

---

## **5. Context Engineering Requirements**
1. Preprocess retrieved context:
   - Chunk selection and overlap for relevance.
   - Maximum token length enforcement.
   - Summarization if context exceeds token limits.
2. Ensure only relevant, authoritative text is included in prompts.
3. Include session memory if needed, without exceeding token budgets.

---

## **6. Prompt Engineering & Guardrails**
1. Build modular prompts:
   - Instruction segment
   - Context segment
   - User query segment
2. Apply **chain-of-thought reasoning** when needed.
3. Inject safety and validation instructions:
   - Check context fidelity
   - Validate output format
   - Provide structured answer schema if required
4. Automatically retry or reformulate prompts if output is invalid.

---

## **7. Performance & Streaming**
1. Stream responses token-by-token using async calls or WebSockets.
2. Ensure minimal latency while respecting context size.
3. Optimize prompt structure for:
   - Shortest effective context
   - Minimal API calls
   - Efficient memory usage
4. Monitor throughput and response times; fallback strategies if slow.

---

## **8. Subsystem Boundaries & Responsibilities**
- FastAPI → Intelligence Layer → Streaming → ChatKit.
- Intelligence Layer is **stateless per request**, except optional session-based memory.
- Do not access databases directly.
- Do not embed text or perform retrieval; only consume pre-fetched context.

---

## **9. Acceptance Criteria**
The subsystem is complete when:

1. It generates accurate, grounded, and contextually coherent responses.
2. It streams responses incrementally to the client.
3. Guardrails and prompt engineering constraints are enforced.
4. Personas are applied dynamically as required.
5. Integrates seamlessly with FastAPI.
6. Logs interactions for auditing and debugging.
7. Maintains high performance and low latency.

---

Output in Chatbot_specs Folder:
- File Type: Markdown
- File Path: Chatbot_specs/agents_sdk_specs/constitution.md

Generate the constitution.md.

## Response snapshot

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

## Outcome

- ✅ Impact: Created comprehensive constitution for Intelligence Layer (OpenAI Agents SDK) Subsystem with all 13 required sections
- 🧪 Tests: null
- 📁 Files: Created Chatbot_specs/agents_sdk_specs/constitution.md
- 🔁 Next prompts: Ready for specification phase (/sp.specify)
- 🧠 Reflection: Successfully defined Intelligence Layer subsystem boundaries, responsibilities, and integration rules

## Evaluation notes (flywheel)

- Failure modes observed: null
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): null