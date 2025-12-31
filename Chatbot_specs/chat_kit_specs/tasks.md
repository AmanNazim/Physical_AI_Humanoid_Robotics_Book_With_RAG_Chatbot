# Tasks: ChatKit UI Subsystem for Global RAG Chatbot System

## 1. Initialization & Environment Setup Tasks

- [ ] T001 Install ChatKit dependencies using uv in chatkit module
- [ ] T002 Create `/chatkit` UI module folder with proper structure
- [ ] T003 Establish file structure for components, contexts, hooks, providers, services in chatkit/
- [ ] T004 Configure linting + formatting for consistent UI structure in chatkit/
- [ ] T005 Prepare environment variables for config endpoint in .env.example
- [ ] T006 Create mock API endpoints for UI testing in chatkit/__mocks__/
- [ ] T007 Add local preview mode for ChatKit integration in chatkit/preview/

## 2. ChatKitProvider Setup Tasks

- [ ] T008 Create ChatKitProvider wrapper in chatkit/providers/ChatKitProvider.jsx
- [ ] T009 Add initialization logic to load config from `/api/config/chatkit` in provider
- [ ] T010 Add streaming support initialization to ChatKitProvider
- [ ] T011 Add global state container (conversation, UI state, errors) to provider
- [ ] T012 Add message buffer for streaming in ChatKitProvider state
- [ ] T013 Add cleanup routines for unmount in ChatKitProvider useEffect
- [ ] T014 Inject provider into Docusaurus layout root in src/theme/Layout/index.js

## 3. Global State & Context Tasks

- [ ] T015 Create ChatUIContext (open/closed state) in chatkit/contexts/ChatUIContext.jsx
- [ ] T016 Create ChatConversationContext in chatkit/contexts/ChatConversationContext.jsx
- [ ] T017 Create hooks: `useChatUI()` in chatkit/hooks/useChatUI.js
- [ ] T018 Create hooks: `useChatMessages()` in chatkit/hooks/useChatMessages.js
- [ ] T019 Create hooks: `useStream()` in chatkit/hooks/useStream.js
- [ ] T020 Implement state transitions for opening/closing chat in context
- [ ] T021 Implement message append logic in ChatConversationContext
- [ ] T022 Implement stream update and finalization logic in context
- [ ] T023 Implement error state handling in global context
- [ ] T024 Add user selection context storage to context provider

## 4. UI Component Implementation Tasks

### 4.1 ChatLauncherButton (Floating Button)

- [ ] T025 Build floating button component in chatkit/components/ChatLauncherButton.jsx
- [ ] T026 Add animations (hover, scale-in, glow) to ChatLauncherButton
- [ ] T027 Add click handler to toggle chat panel in ChatLauncherButton
- [ ] T028 Add mobile-safe hitbox to ChatLauncherButton
- [ ] T029 Ensure correct z-index layering in ChatLauncherButton
- [ ] T030 Add aria-label and accessibility roles to ChatLauncherButton

### 4.2 ChatPanel (Desktop)

- [ ] T031 Build desktop chat container in chatkit/components/ChatPanel.jsx
- [ ] T032 Implement slide/fade animation entry in ChatPanel
- [ ] T033 Add header with close button to ChatPanel
- [ ] T034 Add scroll region with auto-scroll behavior in ChatPanel
- [ ] T035 Add virtualization for long message lists in ChatPanel
- [ ] T036 Add responsive width/height rules to ChatPanel
- [ ] T037 Add dark/light mode compatibility to ChatPanel

### 4.3 MobileChatDrawer

- [ ] T038 Build mobile drawer using full-screen layout in chatkit/components/MobileChatDrawer.jsx
- [ ] T039 Integrate swipe-to-close gesture in MobileChatDrawer
- [ ] T040 Add header with back button to MobileChatDrawer
- [ ] T041 Prevent iOS bounce-scroll issues in MobileChatDrawer
- [ ] T042 Ensure safe-area padding for iPhone in MobileChatDrawer
- [ ] T043 Add animation for open/close in MobileChatDrawer
- [ ] T044 Link to global chat state in MobileChatDrawer

### 4.4 ChatMessageList

- [ ] T045 Implement message list layout in chatkit/components/ChatMessageList.jsx
- [ ] T046 Add timestamp logic to ChatMessageList
- [ ] T047 Add avatars (bot + user) to ChatMessageList
- [ ] T048 Add markdown renderer to ChatMessageList
- [ ] T049 Add code block highlighting to ChatMessageList
- [ ] T050 Add stream placeholder bubble before text finalizes in ChatMessageList
- [ ] T051 Ensure auto-scroll-to-bottom only when user is not scrolled up in ChatMessageList

### 4.5 ChatInputBar

- [ ] T052 Build sticky bottom input bar in chatkit/components/ChatInputBar.jsx
- [ ] T053 Add textarea auto-resize to ChatInputBar
- [ ] T054 Add send button to ChatInputBar
- [ ] T055 Add Enter-to-send logic to ChatInputBar
- [ ] T056 Add debounce for rapid sends to ChatInputBar
- [ ] T057 Add disabled state when streaming is active to ChatInputBar
- [ ] T058 Add attachment hook for future file input (optional) to ChatInputBar

### 4.6 SelectionTooltip

- [ ] T059 Build text-selection detector in chatkit/components/SelectionTooltip.jsx
- [ ] T060 Add tooltip that appears after user highlights book text in SelectionTooltip
- [ ] T061 Add "Ask AI About This" button to SelectionTooltip
- [ ] T062 Add logic to auto-open chat panel in SelectionTooltip
- [ ] T063 Add logic to attach selected_text to message payload in SelectionTooltip
- [ ] T064 Position tooltip relative to DOM selection in SelectionTooltip
- [ ] T065 Add mobile-safe fallback for long-press selection in SelectionTooltip

### 4.7 Markdown Rendering

- [ ] T066 Integrate secure markdown sanitizer in chatkit/components/MarkdownRenderer.jsx
- [ ] T067 Add code highlighting to MarkdownRenderer
- [ ] T068 Add image link blocking (for safety) to MarkdownRenderer
- [ ] T069 Add table support to MarkdownRenderer
- [ ] T070 Add inline math formatting (optional) to MarkdownRenderer
- [ ] T071 Add link preview rules (open in new tab) to MarkdownRenderer
- [ ] T072 Add copy-to-clipboard button for code blocks to MarkdownRenderer

## 5. Backend Integration Tasks

### 5.1 API Connector Layer

- [ ] T073 Create `chatService` with sendMessage() in chatkit/services/chatService.js
- [ ] T074 Create `selectionService` with sendSelectedText() in chatkit/services/selectionService.js
- [ ] T075 Add request cancellation to API connector layer
- [ ] T076 Add timeout logic to API connector layer
- [ ] T077 Add network error detection to API connector layer
- [ ] T078 Add offline fallback message to API connector layer
- [ ] T079 Implement retry logic for failed calls in API connector layer
- [ ] T080 Add telemetry events (optional) to API connector layer

### 5.2 Streaming Handlers

- [ ] T081 Create stream subscription logic in chatkit/services/streamingHandlers.js
- [ ] T082 Parse event-stream responses in streamingHandlers
- [ ] T083 Append tokens into message buffer in streamingHandlers
- [ ] T084 Finalize message when stream completes in streamingHandlers
- [ ] T085 Cancel active stream when new message begins in streamingHandlers
- [ ] T086 Surface stream errors in UI in streamingHandlers

## 6. Integration With Docusaurus Tasks

- [ ] T087 Swizzle Docusaurus Layout component in src/theme/Layout/index.js
- [ ] T088 Insert ChatKitProvider and UI wrapper globally in Layout
- [ ] T089 Add portal root for chat widget in Layout
- [ ] T090 Add global CSS for floating button in chatkit/styles/
- [ ] T091 Lazy-load the widget for performance in Layout
- [ ] T092 Preload on hover in Layout
- [ ] T093 Add route awareness: send page title + chapter ID to backend in Layout
- [ ] T094 Add resets when page changes (optional) in Layout

## 7. Responsiveness & Style Tasks

- [ ] T095 Define responsive breakpoints in chatkit/styles/breakpoints.css
- [ ] T096 Add CSS variables for theme (green theme) in chatkit/styles/variables.css
- [ ] T097 Support dark/light mode sync in chatkit/styles/theme.css
- [ ] T098 Optimize animations for mobile in chatkit/styles/animations.css
- [ ] T099 Add high-contrast mode in chatkit/styles/accessibility.css
- [ ] T100 Add custom scrollbar styling in chatkit/styles/scrollbars.css
- [ ] T101 Create shared styles file in chatkit/styles/shared.css

## 8. Error Handling Tasks

- [ ] T102 Add UI error bubble in chatkit/components/ErrorBubble.jsx
- [ ] T103 Add "retry" action for broken streams in ErrorBubble
- [ ] T104 Add fallback message when API returns non-200 in ErrorBubble
- [ ] T105 Add global fallback UI for system failures in ErrorBubble
- [ ] T106 Add indicator when offline in ErrorBubble
- [ ] T107 Add system messages for boundary failures in ErrorBubble

## 9. Performance Optimization Tasks

- [ ] T108 Implement message virtualization in ChatMessageList
- [ ] T109 Lazy-load heavy components (markdown, code highlighter) in chatkit/components/
- [ ] T110 Debounce user inputs in ChatInputBar
- [ ] T111 Add request deduplication in API connector layer
- [ ] T112 Minimize re-renders using memoized components in chatkit/components/
- [ ] T113 Add Suspense boundaries in chatkit/components/
- [ ] T114 Profile UI performance and tune slow components in chatkit/

## 10. Security & Safety Tasks

- [ ] T115 Sanitize all markdown output in MarkdownRenderer
- [ ] T116 Strip script tags in markdown sanitizer
- [ ] T117 Prevent HTML injection in markdown renderer
- [ ] T118 Never embed API keys in frontend in configuration
- [ ] T119 Implement frontend rate limiting (client-side throttle) in API connector
- [ ] T120 Add UI-level guardrail message if user violates safety rules in chat input
- [ ] T121 Add safe fallback for invalid bot responses in error handling

## 11. QA & Testing Tasks

- [ ] T122 Unit tests for ChatLauncherButton component in chatkit/components/__tests__/ChatLauncherButton.test.js
- [ ] T123 Unit tests for ChatPanel component in chatkit/components/__tests__/ChatPanel.test.js
- [ ] T124 Unit tests for MobileChatDrawer component in chatkit/components/__tests__/MobileChatDrawer.test.js
- [ ] T125 Unit tests for ChatMessageList component in chatkit/components/__tests__/ChatMessageList.test.js
- [ ] T126 Unit tests for ChatInputBar component in chatkit/components/__tests__/ChatInputBar.test.js
- [ ] T127 Unit tests for SelectionTooltip component in chatkit/components/__tests__/SelectionTooltip.test.js
- [ ] T128 Unit tests for MarkdownRenderer component in chatkit/components/__tests__/MarkdownRenderer.test.js
- [ ] T129 Integration tests for streaming in chatkit/__tests__/streaming.test.js
- [ ] T130 Tests for selection tooltip logic in chatkit/__tests__/selection.test.js
- [ ] T131 Tests for mobile responsiveness in chatkit/__tests__/responsive.test.js
- [ ] T132 Tests for light/dark mode in chatkit/__tests__/theme.test.js
- [ ] T133 Test slow network behavior in chatkit/__tests__/network.test.js
- [ ] T134 Simulate server errors in chatkit/__tests__/error.test.js
- [ ] T135 Full E2E test: selection → RAG response → streaming in chatkit/__tests__/e2e.test.js
- [ ] T136 Test with long messages in chatkit/__tests__/message-length.test.js
- [ ] T137 Test conversation reset in chatkit/__tests__/reset.test.js
- [ ] T138 Test cross-page behavior in chatkit/__tests__/navigation.test.js
- [ ] T139 Manual testing on iOS Safari in testing documentation
- [ ] T140 Regression test after integration in CI pipeline

## 12. Documentation Tasks

- [ ] T141 Write component architecture documentation in chatkit/docs/architecture.md
- [ ] T142 Document streaming lifecycle in chatkit/docs/streaming-lifecycle.md
- [ ] T143 Document data flow in chatkit/docs/data-flow.md
- [ ] T144 Document error flow in chatkit/docs/error-flow.md
- [ ] T145 Document integration with Docusaurus in chatkit/docs/docusaurus-integration.md
- [ ] T146 Document UI customization options in chatkit/docs/customization.md
- [ ] T147 Add screenshots/gifs of the UI behavior in chatkit/docs/ui-behavior.md

## 13. Completion Criteria

- [ ] T148 Verify chat opens + closes smoothly in all browsers
- [ ] T149 Verify desktop panel and mobile drawer fully functional
- [ ] T150 Verify selection tooltip works reliably
- [ ] T151 Verify streaming messages render without jitter
- [ ] T152 Verify markdown renderer works safely
- [ ] T153 Verify all API calls and streams function end-to-end
- [ ] T154 Verify fully responsive UI across all screen sizes
- [ ] T155 Verify no blocking errors in error logs
- [ ] T156 Verify all essential tests pass in CI
- [ ] T157 Verify chat integrates cleanly into Docusaurus
- [ ] T158 Verify a real user can select text → ask question → receive streamed reply