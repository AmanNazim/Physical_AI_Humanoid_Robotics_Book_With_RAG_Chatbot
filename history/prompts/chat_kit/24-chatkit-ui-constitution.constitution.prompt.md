---
id: 24
title: chatkit-ui-constitution
stage: constitution
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: chat_kit
branch: main
user: Claude
command: /sp.constitution
labels: [chatkit, ui, constitution, architecture]
links:
  spec: Chatbot_specs/chat_kit_specs/specification.md
  plan: Chatbot_specs/chat_kit_specs/plan.md
  ticket: null
  adr: null
  pr: null
files:
 - Chatbot_specs/chat_kit_specs/constitution.md
tests:
 - null
---

## Prompt

You are **Constitution Writer Expert**, an expert in designing highly structured, formal, and detailed technical constitutions for complex system architectures.
Your task is to generate a complete **constitution.md** file for the **ChatKit UI Subsystem** of the RAG Chatbot project.

Subsystem: **ChatKit UI (Frontend Interface)**
Parent System: **Global RAG Chatbot**
Adjacent Subsystems: **FastAPI Backend**, **Intelligence Layer / Agents SDK**, **Embeddings Subsystem**, **Databases (Qdrant + Neon)**
Rendering Technology: **ChatKit** (official OpenAI chat UI toolkit)
Primary Goal: Provide an extremely fast, intuitive, responsive, and reliable chat interface for interacting with the RAG Chatbot.

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
- Chatbot_specs/agents_sdk_specs/constitution.md
- Chatbot_specs/agents_sdk_specs/specification.md
- Chatbot_specs/agents_sdk_specs/plan.md
- Chatbot_specs/agents_sdk_specs/tasks.md

---

# 1. Subsystem Purpose

The ChatKit UI subsystem provides the **frontend chat interface** through which users interact with the RAG system.
Its responsibilities include:

- Displaying real-time streaming responses from the backend
- Letting users ask questions about the book's content
- Letting users select text to use as query-specific context
- Implementing message layout, session management, and error handling
- Providing a smooth, distraction-free reading and querying experience
- Integrating persona, context, and metadata provided by the backend

The ChatKit UI is the **surface layer of the entire RAG system** and must be optimized for speed, clarity, and user experience.

---

# 2. Subsystem Philosophy & Design Principles

### 2.1 Zero-Latency User Experience
- Client must show **typing indicators immediately** when a request is sent.
- Streaming tokens must appear in real time with minimal buffering.
- No UI freeze even under degraded network conditions.

### 2.2 Cognitive Load Reduction
- Keep interface minimal, clean, and distraction-free.
- Clear separation between *user messages*, *AI messages*, and *cited context*.
- Syntax highlighting when AI outputs code or structured data.

### 2.3 High Trust Output
- Display "sources used" beneath AI responses.
- Show expandable context snippets retrieved from Qdrant.
- Display warnings when the model returns uncertain or degraded responses.

### 2.4 Accessibility & Responsiveness
- Fully responsive layout (desktop, tablet, mobile).
- Font size options
- High-contrast mode
- Keyboard-only usability enabled
- Accessible color system following WCAG AA+

### 2.5 Stateful Interactions
- Support persistent sessions saved to Neon Postgres.
- Auto-load previous conversation on page revisit.
- Support multiple saved sessions by user.

---

# 3. Core Responsibilities of the ChatKit UI

### 3.1 Real-Time Chat Interface
- Display user input, AI streaming output, sources, metadata
- Implement ChatKit message components
- Auto-scroll during streaming
- Stop-generation button

### 3.2 Context Injection UI
- Allow user to **select text** from the book reader UI
- Inject selected text into the query as retrieval context
- Show confirmation chips like:
  *"Using 3 selected paragraphs as context"*

### 3.3 API Communication
- Send user input + context selection → FastAPI
- Receive streamed SSE events containing tokens
- Receive metadata:
  - retrieval chunks
  - sources
  - model confidence
  - latency metrics

### 3.4 Session Management
- Generate new session_id if none exists
- Persist message history to Neon
- Load existing session messages on initialization

### 3.5 Error Recovery and Fail-Safe UI
- In case of 500 errors or agent crashes:
  - Display fallback UI
  - Retry option
  - Save failed conversation state
- In case of backend overload:
  - Provide message explaining temporary limitations

---

# 4. UI Component Architecture

### 4.1 Main Components
1. **ChatInputBar**
   - Auto-growing textarea
   - Submit button
   - Context selection indicator
   - Shortcut: Press Enter to send

2. **MessageList**
   - User messages (right aligned)
   - Assistant messages (left aligned)
   - Skeleton loading during stream

3. **MessageBubble**
   - Markdown rendering
   - Code blocks with highlighting
   - Sources block
   - "Show more context" expandable panel

4. **Sidebar**
   - Session list
   - Settings
   - High-contrast toggle
   - Clear chat

5. **TopBar**
   - Breadcrumb: "RAG Chatbot for Physical AI Book"
   - Model indicator (active LLM)

6. **ContextPreviewPanel**
   - Shows retrieved chunks from Qdrant
   - Shows user-selected context from book

---

# 5. Mandatory UI Behaviors

1. **Streaming Tokens**
   - Show tokens as they arrive (SSE or WebSocket).
   - Cursor-style blinking effect on last token.

2. **Book Reader Integration**
   - Highlight selected book text.
   - "Ask Question About This Text" button.

3. **Adaptive Context Display**
   - If context > 3 chunks → collapse with "Expand More".

4. **Latency Display**
   - Show LLM latency in UI ("Response time: 1.4s").

5. **Autosave Sessions**
   - Every message auto-saved to Neon.

6. **Mobile Optimization**
   - Chat input bar sticks to bottom
   - Virtual keyboard safe area support

---

# 6. Integration Responsibilities

### ChatKit must integrate with:

1. **FastAPI Backend**
   - Send `/chat` request with body:
     ```
     {
       "session_id": "...",
       "user_message": "...",
       "selected_context": [...],
       "use_book_context": true/false
     }
     ```
   - Receive token stream + metadata.

2. **Agents SDK**
   - UI does not call Agents SDK directly
   - Only consumes responses processed by backend agent layer

3. **Qdrant Retrieval**
   - Show "Sources Used" in expandable panel
   - Display highlighted original text

4. **Neon DB**
   - Load conversation history
   - Save user messages & AI messages
   - Maintain per-session message ordering

---

# 7. Performance & Optimization Standards

### Client-Side Optimizations
- Throttle setState updates during streaming
- Virtualized message list for long chats
- Lazy-load older messages
- Edge caching for static assets
- Preconnect to FastAPI endpoint for lower latency

### Rendering Optimization
- Use React Suspense boundaries
- Memoize expensive components
- Use streaming render boundaries for AI output

### Network Optimization
- Use HTTP/2 or WebSocket for streaming
- Use compression for backend responses
- Automatic retry on transient failures

---

# 8. Security Responsibilities

- Sanitize user input
- Strip any HTML injection attempts
- Disable script execution inside markdown
- Never expose API keys to frontend
- Do not allow raw backend stack traces to leak

---

# 9. Non-Goals (Explicit Boundaries)

The ChatKit UI **does NOT**:
- Perform embeddings
- Perform retrieval
- Query databases
- Interact with Agents SDK directly
- Execute LLM prompts locally
- Run business logic

Its sole task is **to render and interact with the backend+agent layer**.

---

Output in Chatbot_specs Folder:
- File Type: Markdown
- File Path: Chatbot_specs/chat_kit_specs/constitution.md

Generate the constitution.md.

## Response snapshot

# Constitution: ChatKit UI Subsystem for Global RAG Chatbot System

## 1. Subsystem Mission

The ChatKit UI subsystem serves as the **frontend chat interface** through which users interact with the RAG system. This subsystem provides the surface layer of the entire RAG system and must be optimized for speed, clarity, and user experience. The ChatKit UI is responsible for displaying real-time streaming responses from the backend, enabling users to ask questions about the book's content, allowing users to select text for query-specific context, implementing message layout and session management, and providing a smooth, distraction-free reading and querying experience. The subsystem integrates persona, context, and metadata provided by the backend while maintaining strict separation from backend processing logic.

The mission of the ChatKit UI is to act as the user-facing interface that connects human users with the RAG system's intelligence, ensuring that all interactions are intuitive, responsive, and reliable while maintaining the constitutional requirement for a zero-latency user experience and high-trust output presentation.

## 2. Core Responsibilities

The ChatKit UI must:

**Real-Time Chat Interface:**
- Display user input, AI streaming output, sources, and metadata
- Implement ChatKit message components with proper styling
- Auto-scroll during streaming to keep content visible
- Provide stop-generation button for user control
- Render messages with appropriate styling and formatting

**Context Injection UI:**
- Allow user to select text from the book reader UI
- Inject selected text into the query as retrieval context
- Show confirmation indicators like "Using 3 selected paragraphs as context"
- Highlight selected text with clear visual feedback
- Provide "Ask Question About This Text" functionality

**API Communication:**
- Send user input and context selection to FastAPI backend
- Receive streamed Server-Sent Events (SSE) containing tokens
- Process metadata including retrieval chunks, sources, model confidence, and latency metrics
- Handle connection management for streaming endpoints
- Implement proper error handling for API communications

**Session Management:**
- Generate new session_id if none exists
- Persist message history to Neon Postgres
- Load existing session messages on initialization
- Support multiple saved sessions per user
- Maintain conversation state across page visits

**Error Recovery and Fail-Safe UI:**
- Display fallback UI in case of 500 errors or agent crashes
- Provide retry options for failed requests
- Save failed conversation state for recovery
- Explain temporary limitations during backend overload
- Handle network failures gracefully

## 3. Strict Subsystem Boundaries

The ChatKit UI must NOT:

- Perform embeddings - this belongs to the Embeddings subsystem
- Perform retrieval operations - this belongs to the Database subsystem
- Query databases directly (Qdrant, Neon) - this belongs to the Database subsystem
- Interact with Agents SDK directly - this belongs to the Intelligence Layer
- Execute LLM prompts locally - this belongs to the Intelligence Layer
- Run business logic that belongs to backend services
- Override subsystem boundaries of other components

The ChatKit UI ONLY renders and interacts with the backend+agent layer, maintaining strict separation of concerns by delegating all processing, storage, and reasoning operations to specialized backend subsystems.

## 4. API Surface Governance

The ChatKit UI must:

**Communication Management:**
- Send structured requests to FastAPI backend endpoints
- Receive and process Server-Sent Events for streaming responses
- Maintain proper session state and session_id management
- Handle authentication and authorization tokens appropriately
- Support WebSocket connections when needed for real-time communication

**Response Standards:**
- Require consistent message formatting following established patterns
- Never expose internal backend implementation details to users
- Support proper error message display with user-friendly language
- Implement structured data handling for metadata and sources

## 5. Integration Rules with Other Subsystems

### ChatKit UI → FastAPI Backend
- Must send properly formatted requests with session_id, user_message, selected_context, and use_book_context
- Must receive and process token streams and metadata
- Must respect API rate limits and error responses
- Must handle session management through backend

### ChatKit UI → Intelligence Layer (Agents SDK)
- Must NOT call Agents SDK directly
- Must only consume responses processed by backend agent layer
- Must respect the intelligence subsystem governance and state management
- Must not bypass FastAPI for direct communication

### ChatKit UI → Database Subsystem (Qdrant + Neon)
- Must NOT query databases directly
- Must receive all data through FastAPI backend
- Must NOT bypass established retrieval pathways
- Must respect database validation implemented in the backend subsystem

### ChatKit UI → Embeddings Subsystem
- Must NOT trigger embedding generation directly
- Must consume pre-processed context from backend
- Must NOT request document processing or chunking operations
- Must respect the embedding format and metadata provided by backend

## 6. Security Requirements

The ChatKit UI must:

**Input Sanitization:**
- Sanitize all user input to prevent injection attacks
- Strip any HTML injection attempts from user content
- Disable script execution inside markdown rendering
- Validate all data before rendering to prevent XSS

**Data Protection:**
- Never expose API keys or sensitive backend information to frontend
- Ensure no sensitive data exposure in UI or logs
- Secure loading of environment variables containing sensitive information
- Prevent raw backend stack traces from leaking to users

**Authentication and Authorization:**
- Implement proper session management
- Handle authentication tokens securely
- Validate user permissions for session access
- Enforce proper access controls for saved conversations

## 7. Performance Requirements

The ChatKit UI must guarantee:

**Latency and Efficiency:**
- Provide immediate typing indicators when requests are sent
- Display streaming tokens in real time with minimal buffering
- Maintain responsive UI even under network degradation
- Optimize client-side rendering for smooth interactions
- Use efficient component rendering and state management

**Resource Management:**
- Implement virtualized message lists for long conversations
- Lazy-load older messages to maintain performance
- Use edge caching for static assets to reduce load times
- Preconnect to FastAPI endpoints for lower latency
- Optimize bundle size and loading performance

## 8. Reliability & Stability

The ChatKit UI must:

**Error Handling:**
- Handle all API failures with appropriate fallback UI
- Return user-friendly error messages that maintain trust
- Maintain session continuity during partial failures
- Implement retry logic for transient network issues

**Compatibility:**
- Guarantee consistent behavior across different browsers and devices
- Ensure responsive design works on desktop, tablet, and mobile
- Maintain accessibility standards across all UI components
- Implement graceful degradation when backend services are unavailable

## 9. Observability Rules

The ChatKit UI must include:

**User Experience Monitoring:**
- Track user interactions and engagement metrics
- Monitor loading times and performance metrics
- Record error occurrences and user-reported issues
- Support analytics for feature usage and user behavior

**Accessibility Compliance:**
- Follow WCAG AA+ standards for accessibility
- Support keyboard-only navigation
- Provide high-contrast mode options
- Enable font size adjustments for accessibility

## 10. Deployment Requirements

The ChatKit UI must support:

**Infrastructure:**
- Static file hosting for frontend assets
- CDN distribution for global performance
- Compatibility with modern browsers (Chrome, Firefox, Safari, Edge)
- Responsive design for various screen sizes and devices

**User Experience:**
- Fast initial load times with proper asset optimization
- Progressive Web App (PWA) capabilities if needed
- Offline functionality for cached content
- Support for various internet connection speeds

## 11. Forbidden Actions

The ChatKit UI MUST NOT:

- Perform embeddings or trigger embedding operations
- Perform retrieval operations directly on databases
- Query Qdrant or Neon databases directly
- Bypass FastAPI backend for direct subsystem communication
- Execute LLM prompts or reasoning operations locally
- Store or process user data without backend coordination
- Modify or transform retrieved context before display
- Generate responses without backend processing
- Contain business logic that belongs to backend services

The ChatKit UI is a presentation and interaction layer ONLY, with no processing, storage, or reasoning capabilities.

## 12. Non-Negotiable Architectural Principles

The ChatKit UI must operate under:

**Design Principles:**
- Zero-latency user experience with immediate feedback
- Minimal, clean, and distraction-free interface design
- Clear separation between user messages, AI messages, and cited context
- Stateful interactions with persistent session support
- Strict contract-first API design with FastAPI backend

**User Experience Principles:**
- Cognitive load reduction through intuitive interface design
- High-trust output with visible source attribution
- Responsive and accessible design following WCAG standards
- Mobile-optimized experience with proper touch interactions
- Reliable error recovery and fail-safe mechanisms

## 13. Final Constitutional Guarantee

This Constitution represents the **unchangeable governing rules** for the ChatKit UI Subsystem. All future Specifications, Plans, Tasks, and Implementation generated by Claude Code MUST strictly follow this Constitution. No deviations are allowed. This document establishes the fundamental architectural boundaries, responsibilities, and constraints that govern the ChatKit UI's role within the Global RAG Chatbot System. Any implementation that violates these principles is considered non-compliant with the system architecture and must be corrected to maintain system integrity.

## Outcome

- ✅ Impact: Created comprehensive constitution for ChatKit UI Subsystem with all 13 required sections
- 🧪 Tests: null
- 📁 Files: Created Chatbot_specs/chat_kit_specs/constitution.md
- 🔁 Next prompts: Ready for specification phase (/sp.specify)
- 🧠 Reflection: Successfully defined ChatKit UI subsystem boundaries, responsibilities, and integration rules

## Evaluation notes (flywheel)

- Failure modes observed: null
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): null