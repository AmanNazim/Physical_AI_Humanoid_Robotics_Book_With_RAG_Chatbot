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