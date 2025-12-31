# Specification: ChatKit UI Subsystem for Global RAG Chatbot System

## 1. Purpose of This Specification

This document defines the complete UI specification for integrating a **ChatKit-powered chatbot widget** inside the **Docusaurus book website**, including a floating chat-launcher button, a responsive, animated chat panel, full ChatKit configuration, backend connectivity, mobile & desktop responsive layout, security, performance, and optimization requirements, theming (matching the green theme of the book), accessibility rules (WCAG AA), streaming UX behavior, error handling display, and user text selection → send to chatbot pipeline (UI side).

## 2. User Experience Requirements

### 2.1 Floating Chat Button (Launcher)
- Always visible at bottom-right corner
- Button shape: **circular**, elevated shadow, soft glow on hover
- Color: **Green theme (#00C26A or book theme)**
- Icon: Message / Chat bubble (Lucide or custom SVG)
- Gesture behavior:
  - **Click → opens chat panel**
  - **Second click → closes chat panel**
- Animation:
  - Scale-in on hover
  - Smooth "slide from bottom-right" when appearing
- Accessible for mobile & desktop

### 2.2 Desktop Chat Panel Specifications
- Position: **Bottom-right**, floating modal drawer
- Width: **420px**
- Height: **70vh**
- Border radius: **16px**
- Padding: **16px**
- Glass-morphism OR soft shadow UI
- Must not block main reading content
- Must auto-resize if screen height is small
- Close button inside top-right
- Conversation scrollable area with auto-scroll

### 2.3 Mobile Chat Panel Specifications
- Open as a **full-screen drawer**
- Height: **100vh**
- Width: **100vw**
- Slide-up animation from bottom
- Floating launcher still stays outside
- Back button instead of close button
- Input bar is sticky at the bottom
- Must never overflow iOS safe area (padding-bottom: env(safe-area-inset-bottom))

### 2.4 Responsiveness Rules
- < 768px width → Mobile mode
- >= 768px → Desktop mode
- Fluid scaling
- Use CSS variables for theme colors
- Minimum tap target = 44px

## 3. ChatKit Integration Requirements

### 3.1 ChatKit SDK Setup
- Must initialize ChatKit with:
  - **OpenRouter API Key** (provided from backend via session token, not hardcoded)
  - Chat endpoint: `/api/agent/chat` from FastAPI
  - Streaming enabled
- Must support:
  - Markdown rendering
  - Code highlighting
  - Avatar bubble for bot and user
  - Loading indicators
  - Retry on failed messages

### 3.2 ChatKit Event Model
- Use hooks/events for:
  - onSend → call FastAPI route
  - onStreamChunk → update UI in real time
  - onError → show error bubble
- Must show "Agent is thinking…" indicator for slow retrieval calls

## 4. Connection to Backend (FastAPI)

### 4.1 Required API Routes
- `POST /api/agent/chat` → send user message, receive streamed response
- `POST /api/rag/selection` → send highlighted text from book
- `GET /api/config/chatkit` → retrieve safe client config
- `GET /api/health` → health check for uptime UI indicator

### 4.2 Payload Specifications
ChatKit must send JSON payload:
```
{
"message": "<user_message>",
"context": {
"selected_text": "<optional_highlighted_text>",
"page": "<book_page_or_chapter>"
}
}
```

### 4.3 Streaming
- Must render tokens as they stream
- Must keep scroll pinned to bottom
- Must show "---" separator when streaming ends
- Must gracefully reconnect on network break

## 5. Docusaurus Integration Requirements

### 5.1 Mounting Strategy
- ChatKit widget must be mounted once, at the root of the Docusaurus layout
- Use `swizzled` Docusaurus layout components for injection:
  - `/src/theme/Layout/index.js`
- The floating button must overlay the entire site including docs pages

### 5.2 Chat Button Placement
- Fixed at bottom-right
- z-index: 999999
- Works on all pages including docs, blog, and custom pages

### 5.3 Book Page Text Selection → RAG Query
- When a user highlights text in the book:
  - A small "Ask AI about this" tooltip must appear
  - On click, this content is passed as `selected_text` to the backend
- Tooltip:
  - Position: near selection
  - Color: green theme
  - Good contrast and shadow

## 6. UI Component Breakdown (Detailed)

### 6.1 ChatLauncherButton
- Circular button with green theme color
- Message/chat bubble icon
- Hover and active states with animations
- Proper accessibility attributes (aria-label, role)

### 6.2 ChatPanelContainer
- Responsive container with fixed positioning
- Desktop: 420px width, 70vh height
- Mobile: Full screen dimensions
- Glass-morphism or soft shadow styling
- Close/back button functionality

### 6.3 ChatHeader
- Title section with chatbot identification
- Close/back button for mobile/desktop
- Health status indicator for backend connection

### 6.4 ChatMessageList
- Scrollable container for messages
- Auto-scroll to bottom when new messages arrive
- Proper styling for user and bot messages

### 6.5 ChatMessageBubble (user/bot variants)
- Different styling for user vs bot messages
- Avatar bubbles for both user and bot
- Markdown rendering support
- Code block highlighting
- Source citation display

### 6.6 ChatInputBar
- Text input area with auto-grow functionality
- Send button
- Context selection indicator
- Keyboard shortcuts support

### 6.7 MobileChatDrawer
- Full-screen drawer component for mobile
- Slide-up animation from bottom
- Proper iOS safe area handling
- Back button instead of close button

### 6.8 SelectionTooltip
- Tooltip that appears when text is selected
- Green theme styling
- "Ask AI about this" text
- Proper positioning near selection

### 6.9 ChatKitProvider
- Context provider for ChatKit state management
- API key and configuration management
- Event handling and state synchronization

### 6.10 MarkdownRenderer
- Secure rendering of markdown content
- Code block syntax highlighting
- Link handling and security
- Image rendering support

### 6.11 LoadingSkeleton / TypingIndicator
- Visual indicators for loading states
- Typing animation for bot responses
- Smooth transitions between states

## 7. Accessibility Requirements

- WCAG 2.1 AA compliance
- Proper aria-labels for buttons
- Escape key closes chat
- Screen reader announcements for streaming updates
- Keyboard navigation support for all interactive elements
- Sufficient color contrast ratios
- Focus indicators for interactive elements
- Proper semantic HTML structure

## 8. Performance Requirements

- Lazy-load ChatKit UI bundle
- Use React Suspense for loading states
- Minimize render cycles during streaming
- Preload only when user hovers launcher
- Use Framer Motion for smooth animations
- Optimize bundle size and loading performance
- Efficient virtual scrolling for long message histories
- Debounce/throttle for real-time updates during streaming

## 9. Security Requirements

- No API keys in frontend ever
- ChatKit only uses backend-issued temporary tokens
- Prevent XSS in user input and bot Markdown
- Sanitize HTML output
- Rate limit user messages (frontend throttling)
- Input validation and sanitization
- Secure communication with backend endpoints
- Proper CORS configuration for API requests

## 10. Theming Requirements

- Green theme, soft modern appearance
- Rounded corners everywhere
- Light/Dark mode sync with Docusaurus theme
- Shadows = subtle, non-intrusive
- Consistent color palette across all components
- CSS variables for easy theme customization
- Responsive typography that scales appropriately
- Consistent spacing and layout patterns