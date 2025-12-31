---
id: 26
title: chatkit-ui-implementation-plan
stage: plan
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: chat_kit
branch: main
user: Claude
command: /sp.plan
labels: [chatkit, ui, plan, architecture]
links:
  constitution: Chatbot_specs/chat_kit_specs/constitution.md
  spec: Chatbot_specs/chat_kit_specs/specification.md
  ticket: null
  adr: null
  pr: null
files:
 - Chatbot_specs/chat_kit_specs/plan.md
tests:
 - null
---

## Prompt

You are **Planning Architect Expert**, an expert in translating detailed specifications into fully actionable development plans.

Subsystem: ChatKit UI – RAG Chatbot User Interface
Parent System: RAG Chatbot for Physical AI Humanoid Robotics Book

Your task:
Generate a **full, deeply detailed plan.md** for the ChatKit UI subsystem, strictly adhering to the Constitution + Specification previously defined.

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
- Chatbot_specs/chat_kit_specs/constitution.md (important)
- Chatbot_specs/chat_kit_specs/specification.md (important)

The plan must describe EXACTLY:
- How the system will be implemented
- How components interact
- Step-by-step execution order
- Dependencies
- Integration flow with backend + Docusaurus
- UI component architecture
- State management strategy
- Data flow strategy (Frontend ↔ FastAPI ↔ Agent SDK)
- Streaming pipeline
- Error recovery
- Performance optimization

NO code. ONLY actionable technical planning.

---------------------------------------------------------------------

# 1. STRUCTURE OF THE PLAN

Your plan.md MUST contain the following structured sections:

## 1. Overview
A clear description of:
- The UI subsystem purpose
- How ChatKit will handle the user experience
- How it supports the entire RAG system
- What is included and excluded in the UI subsystem implementation

## 2. Architectural Breakdown
Describe the architecture including:
- ChatLauncherButton
- ChatPanel (desktop)
- MobileChatDrawer
- ChatMessageList
- ChatInputBar
- StreamingRenderer (token-by-token UI updates)
- Markdown renderer
- SelectionTooltip integration
- ChatKitProvider configuration
- Connection layer: where and how API endpoints are called
- Global event layers (ChatKit streams, error states, typing indicator)

Show how each component depends on others.

## 3. Data Flow Plan
FULL pipeline description:

### 3.1 User message flow
1. User enters message
2. ChatInputBar triggers ChatKit’s send()
3. Payload structure with selected_text + context
4. FastAPI receives
5. Agent SDK processes
6. Qdrant retrieval (if needed)
7. Stream starts
8. ChatKit UI renders tokens
9. Final response displayed

### 3.2 Selected text flow
1. User highlights text
2. SelectionTooltip appears
3. User clicks "Ask AI About This"
4. UI packages selected text
5. Sends to API
6. Chat opens automatically
7. Streaming response returned

### 3.3 Error flow
- API unreachable
- Streaming breaks
- Timeout handling
- UI must show retry options
- Must append system message "⚠️ Something went wrong."

## 4. Component Implementation Plan
Explain in DETAIL how each component will be built and interact:

### 4.1 ChatKitProvider Setup
- Load safe config from `/api/config/chatkit`
- Initialize ChatKit with streaming enabled
- Store authentication token and refresh if expired
- Provide global context

### 4.2 Launcher Button
- Create floating button
- Add animations (scale, hover glow)
- Toggle global ChatUI state

### 4.3 Chat Panel (Desktop)
- Dimensions
- Animation logic
- Scroll management
- Message list virtualization

### 4.4 Mobile Drawer
- Full-screen drawer
- iOS safe-area adjustments
- Back button
- Prevent overscroll

### 4.5 Chat Message Rendering
- Markdown → sanitized
- Code blocks highlighted
- Bot avatar + user avatar
- Stream chunk rendering

### 4.6 Input System
- Sticky bottom bar
- Enter-to-send
- Throttle quick sends
- Prevent empty messages

### 4.7 Selection Tooltip
- Appears on highlight
- Positioned relative to selection range
- Triggers RAG message with context

### 4.8 Theming
- CSS variables
- Light/dark mode sync
- Green theme surface

### 4.9 Accessibility
- Keyboard support
- Screen reader announcements
- ARIA roles
- High contrast mode

### 4.10 Error Components
- Error bubble
- Retry button
- Network loss detection

## 5. Integration Plan with Docusaurus
Detail exactly how the ChatKit UI will integrate:

### 5.1 Injection Strategy
- Swizzle Layout
- Inject ChatKit UI root container
- Mount ChatKitProvider globally

### 5.2 Script Loading
- Lazy load the chat widget component
- Preload on hover

### 5.3 Chat Button Placement
- Fixed bottom-right via global CSS
- z-index rules
- Works across docs + blog + custom pages

### 5.4 Routing Awareness
- Detect page + chapter
- Pass as context to backend
- Reset conversation on page change (configurable)

## 6. Backend Integration Plan
Define clearly:

- `/api/agent/chat` for messages
- `/api/rag/selection` for selected text
- `/api/config/chatkit` for safe ChatKit config
- Streaming handling rules
- Timeout strategy
- Safety fallback content

## 7. Performance Optimization Strategy
Include:

- Lazy-loading ChatKit bundle
- Debounce input
- Request cancellation
- Token streaming virtualization
- React rendering optimization

## 8. Security Plan
- No API keys in frontend
- Using short-lived temp tokens
- HTML sanitization
- Rate limit user sends
- XSS/Injection prevention

## 9. Testing Plan
Detail all test categories:
- UI integration tests
- Streaming test cases
- Tooltip interaction
- Text input tests
- Docusaurus routing tests
- Mobile device tests

## 10. Completion Criteria
Define exact “done” requirements for the subsystem:
- UI fully functional
- Streaming works smoothly
- Tooltip integrated
- Config loading works
- End-to-end connectivity to FastAPI functional
- Mobile + desktop responsive
- No blocking UI bugs
- Accessibility verified

---------------------------------------------------------------------

# 2. OUTPUT RULES

- Output must be **plan.md** only
- No explanations
- No code
- No implementation details
- Pure planning
- Must be actionable and exhaustive
- Must follow Constitution + Specification perfectly

---------------------------------------------------------------------

Output in Chatbot_specs Folder:
- File Type: Markdown
- File Path: Chatbot_specs/chat_kit_specs/plan.md.md

Generate the **plan.md** now.

## Response snapshot

# Implementation Plan: ChatKit UI Subsystem for Global RAG Chatbot System

## 1. Overview

The UI subsystem serves as the frontend chat interface through which users interact with the RAG system. ChatKit will handle the complete user experience including real-time streaming responses, text selection for context injection, message layout, session management, and error handling. The subsystem supports the entire RAG system by providing an intuitive, responsive, and reliable chat interface that allows users to ask questions about the book's content and receive contextual responses. The UI subsystem implementation includes all components for user interaction, message rendering, and backend communication, while excluding any backend processing, embedding generation, database queries, or LLM reasoning capabilities.

## 2. Architectural Breakdown

The architecture includes the following components with their dependencies:

- **ChatLauncherButton**: Floating button that toggles the chat panel; depends on global UI state management
- **ChatPanel (desktop)**: Modal drawer with fixed dimensions; depends on ChatLauncherButton state and message list
- **MobileChatDrawer**: Full-screen drawer for mobile; depends on device detection and global UI state
- **ChatMessageList**: Scrollable container for messages; depends on message history and streaming updates
- **ChatInputBar**: Input area with submission controls; depends on message sending functionality
- **StreamingRenderer**: Token-by-token UI updates; depends on streaming data from backend
- **Markdown renderer**: Content formatting; depends on message content
- **SelectionTooltip**: Text selection integration; depends on document selection APIs
- **ChatKitProvider**: Global context provider; depends on backend configuration
- **Connection layer**: API endpoint communication; depends on ChatKitProvider configuration
- **Global event layers**: Stream, error, and typing indicator management; depend on all UI components

## 3. Data Flow Plan

### 3.1 User message flow
1. User enters message in ChatInputBar
2. ChatInputBar triggers ChatKit's send() function
3. Payload structure includes selected_text + context with session_id
4. FastAPI receives the request at `/api/agent/chat`
5. Agent SDK processes the request with provided context
6. Qdrant retrieval occurs if needed for additional context
7. Stream starts from backend to frontend
8. ChatKit UI renders tokens as they arrive
9. Final response is displayed with sources and metadata

### 3.2 Selected text flow
1. User highlights text in the book content
2. SelectionTooltip appears near the selection
3. User clicks "Ask AI About This" button
4. UI packages selected text as context
5. Sends request to API with selected_text parameter
6. Chat panel opens automatically if closed
7. Streaming response is returned and displayed

### 3.3 Error flow
- API becomes unreachable → UI shows connection error state
- Streaming breaks → UI shows retry option and caches unsent message
- Timeout handling → UI shows timeout warning with retry option
- UI shows retry options for failed requests
- System appends message "⚠️ Something went wrong." when errors occur

## 4. Component Implementation Plan

### 4.1 ChatKitProvider Setup
- Load safe config from `/api/config/chatkit` endpoint
- Initialize ChatKit with streaming enabled and proper API key management
- Store authentication token and implement refresh mechanism when expired
- Provide global context for all ChatKit components

### 4.2 Launcher Button
- Create floating circular button with green theme
- Add hover animations (scale-in, glow effect)
- Toggle global ChatUI state between open/closed
- Implement proper z-index stacking for visibility

### 4.3 Chat Panel (Desktop)
- Implement fixed dimensions: 420px width, 70vh height
- Add slide-in/out animation from bottom-right
- Implement auto-scroll management for new messages
- Include virtualization for long message histories

### 4.4 Mobile Drawer
- Create full-screen overlay drawer
- Implement slide-up/down animation
- Add iOS safe-area padding adjustments
- Include back/close button functionality
- Prevent overscroll behavior

### 4.5 Chat Message Rendering
- Implement sanitized markdown rendering
- Add syntax highlighting for code blocks
- Create distinct avatar styling for bot and user messages
- Handle stream chunk rendering with cursor indicators

### 4.6 Input System
- Create sticky bottom input bar
- Implement Enter-key submission handling
- Add throttling for rapid message sending
- Prevent submission of empty messages
- Include character counting and validation

### 4.7 Selection Tooltip
- Detect text selection events on document
- Position tooltip near selection range with proper offset
- Style with green theme and appropriate shadows
- Trigger RAG message flow with selected text as context

### 4.8 Theming
- Define CSS variables for green theme colors
- Implement light/dark mode detection and sync
- Apply rounded corner styling consistently
- Add subtle shadow effects for depth

### 4.9 Accessibility
- Implement keyboard navigation support
- Add screen reader announcements for streaming updates
- Apply proper ARIA roles and labels
- Include high contrast mode toggle functionality

### 4.10 Error Components
- Create error message bubble display
- Implement retry button functionality
- Add network loss detection indicators
- Show appropriate error messages based on error type

## 5. Integration Plan with Docusaurus

### 5.1 Injection Strategy
- Swizzle the main Docusaurus Layout component
- Inject ChatKit UI root container at document level
- Mount ChatKitProvider globally to ensure availability
- Ensure injection happens after DOM is ready

### 5.2 Script Loading
- Implement lazy loading for the chat widget component bundle
- Add preload functionality when user hovers launcher button
- Optimize loading order to prevent blocking main content
- Implement service worker caching for faster subsequent loads

### 5.3 Chat Button Placement
- Apply fixed positioning at bottom-right of viewport
- Set z-index to 999999 to ensure visibility above all content
- Implement responsive positioning for different screen sizes
- Ensure button works across docs, blog, and custom pages

### 5.4 Routing Awareness
- Detect current page and chapter information
- Pass page context as metadata in API requests
- Implement optional conversation reset on page changes
- Maintain session continuity across related pages

## 6. Backend Integration Plan

### 6.1 API Endpoints
- `/api/agent/chat`: Handle user messages and streaming responses
- `/api/rag/selection`: Process selected text queries
- `/api/config/chatkit`: Deliver safe client configuration
- Implement proper request/response validation

### 6.2 Streaming Handling
- Establish Server-Sent Events (SSE) or WebSocket connection
- Implement token-by-token rendering with proper buffering
- Handle connection interruptions and automatic reconnection
- Manage stream termination and final response assembly

### 6.3 Timeout Strategy
- Set appropriate request timeouts (default 30 seconds)
- Implement progressive timeout increases for long-running requests
- Provide user feedback during extended processing times
- Gracefully handle timeout scenarios with retry options

### 6.4 Safety Fallback Content
- Implement default responses for API failures
- Create fallback UI states for different error scenarios
- Provide helpful messaging when backend services are unavailable
- Maintain conversation context during temporary outages

## 7. Performance Optimization Strategy

### 7.1 Bundle Optimization
- Implement code splitting for ChatKit components
- Lazy-load heavy dependencies only when needed
- Optimize bundle size through tree-shaking
- Implement caching strategies for static assets

### 7.2 Input Handling
- Add debouncing to prevent excessive API calls
- Implement smart message batching when appropriate
- Optimize rendering during streaming with React.memo
- Use virtualization for long message histories

### 7.3 Request Management
- Implement request cancellation for interrupted streams
- Add proper loading states to prevent UI jank
- Optimize API call frequency and batching
- Cache configuration and static data appropriately

### 7.4 Rendering Optimization
- Use React Suspense for loading boundaries
- Implement proper component memoization
- Optimize token streaming rendering performance
- Virtualize message lists for improved scrolling

## 8. Security Plan

### 8.1 API Key Management
- Never store permanent API keys in frontend code
- Use short-lived temporary tokens from backend
- Implement automatic token refresh mechanisms
- Secure token storage using appropriate browser APIs

### 8.2 Content Sanitization
- Sanitize all user-generated content before display
- Implement HTML sanitization for markdown rendering
- Prevent XSS attacks through proper input validation
- Use Content Security Policy headers appropriately

### 8.3 Rate Limiting
- Implement frontend throttling for message sending
- Add cooldown periods between consecutive requests
- Show appropriate feedback when rate limits are reached
- Coordinate with backend rate limiting strategies

### 8.4 Injection Prevention
- Validate and sanitize all API request payloads
- Implement proper output encoding for all responses
- Prevent injection attacks through proper input handling
- Use secure communication protocols (HTTPS)

## 9. Testing Plan

### 9.1 UI Integration Tests
- Test component interactions and state management
- Validate responsive behavior across different screen sizes
- Verify accessibility features and keyboard navigation
- Check proper rendering of various message types

### 9.2 Streaming Tests
- Test token-by-token rendering during streaming
- Validate stream interruption and recovery
- Verify proper handling of slow and fast responses
- Check stream termination and final message assembly

### 9.3 Interaction Tests
- Test tooltip appearance and positioning
- Validate text selection and context injection
- Check launcher button functionality and animations
- Verify input handling and submission processes

### 9.4 Docusaurus Integration Tests
- Test functionality across different page types (docs, blog, custom)
- Validate proper injection and initialization
- Check routing awareness and context passing
- Verify compatibility with Docusaurus themes and plugins

### 9.5 Device and Browser Tests
- Test mobile and desktop responsiveness
- Validate functionality across different browsers
- Check touch and mouse interaction support
- Verify iOS safe area and Android compatibility

## 10. Completion Criteria

The subsystem is complete when:

- UI is fully functional with all components working properly
- Streaming works smoothly with real-time token rendering
- Selection tooltip is integrated and functional
- Configuration loading works reliably from backend
- End-to-end connectivity to FastAPI is established and tested
- Mobile and desktop layouts are responsive and optimized
- No blocking UI bugs exist in critical functionality
- Accessibility requirements are verified and met
- All integration points with Docusaurus are working
- Error handling and recovery mechanisms are in place
- Performance meets specified optimization targets
- Security measures are properly implemented and tested

## Outcome

- ✅ Impact: Created comprehensive implementation plan for ChatKit UI Subsystem with all 10 required sections
- 🧪 Tests: null
- 📁 Files: Created Chatbot_specs/chat_kit_specs/plan.md
- 🔁 Next prompts: Ready for tasks generation phase (/sp.tasks)
- 🧠 Reflection: Successfully defined ChatKit UI subsystem architecture, integration plan, and completion criteria

## Evaluation notes (flywheel)

- Failure modes observed: null
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): null