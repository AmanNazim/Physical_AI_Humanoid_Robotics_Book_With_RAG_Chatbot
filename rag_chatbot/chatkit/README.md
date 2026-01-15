# ChatKit UI Subsystem

The ChatKit UI Subsystem is a comprehensive chat interface designed for the Humanoid Robotics RAG Chatbot. It provides a premium, modern, and highly usable UI/UX with full Docusaurus integration.

## Features

- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Real-time Streaming**: Streaming responses from the AI backend
- **Text Selection Integration**: Ability to ask questions about selected text
- **Accessibility**: WCAG AA compliant with keyboard navigation
- **Theming**: Supports light/dark themes with system preference detection
- **Docusaurus Integration**: Seamlessly integrates with Docusaurus documentation sites
- **Secure Markdown Rendering**: Sanitized markdown with syntax highlighting
- **Loading States**: Visual feedback during API calls and streaming
- **Error Handling**: Graceful error handling with retry mechanisms

## Components

### Core Components
- `ChatKitProvider` - Main provider component managing global chat state
- `ChatUIContext` - Context for managing UI state (open/closed, mobile/desktop)
- `ChatConversationContext` - Context for managing conversation state
- `ChatLauncherButton` - Floating button component that triggers chat panel
- `ChatPanel` - Desktop chat panel with fixed dimensions
- `MobileChatDrawer` - Full-screen mobile drawer for mobile devices

### UI Components
- `ChatMessageList` - Scrollable container for messages with auto-scroll
- `ChatMessageBubble` - Individual message display with different styling for user/bot
- `MarkdownRenderer` - Secure markdown rendering with syntax highlighting
- `ChatInputBar` - Sticky input bar with textarea and send button
- `SelectionTooltip` - Tooltip that appears when text is selected
- `ErrorBubble` - Component for displaying error messages with retry option

### Service Components
- `chatService.js` - API service layer for communication with backend
- `streamingHandlers.js` - Handlers for processing streaming responses

## API Endpoints

The ChatKit UI connects to the following backend API endpoints:

- `GET /api/v1/config/` - Get configuration for ChatKit UI
- `POST /api/v1/chat` - Send a message to the chat API
- `POST /api/v1/chat/stream` - Send a message with streaming response
- `GET /api/v1/health` - Check the health of the backend

## Integration

The ChatKit UI is integrated into the Docusaurus Layout component and provides:

1. A floating "Ask AI" button on all pages
2. A slide-up drawer on mobile devices
3. A fixed panel on desktop devices
4. Seamless integration with the site's theme
5. Text selection integration for contextual questions

## Styling

The ChatKit UI uses a CSS variable system for theming:

- `variables.css` - CSS variables defining the green theme and other design tokens
- `theme.css` - Theme definitions for light/dark mode
- `breakpoints.css` - Responsive breakpoints for mobile/desktop layouts
- `animations.css` - CSS animations for UI transitions and interactions

## Security

- Markdown content is sanitized using `rehype-sanitize`
- Links open in new tabs with `noopener noreferrer`
- All API calls use proper CORS handling
- Content Security Policy friendly
- Input validation and sanitization

## Performance

- Efficient state management with React Context
- Optimized rendering with React.memo where appropriate
- Streaming responses for real-time updates
- Lazy loading for components when needed
- Minimal DOM updates during streaming

## Accessibility

- Full keyboard navigation support
- Proper ARIA attributes
- Screen reader friendly
- WCAG AA compliant color contrast
- Focus management
- Semantic HTML structure

## Development

To run the ChatKit UI in development mode:

1. Ensure the backend API is running
2. The UI will automatically connect to the backend endpoints
3. Use the Docusaurus development server to see changes

## Architecture

The ChatKit subsystem follows a component-based architecture with:

- **Context Providers**: Global state management for UI and conversations
- **Custom Hooks**: Encapsulated logic for different aspects of the chat
- **UI Components**: Reusable components for the chat interface
- **Services**: API communication and streaming handlers
- **Styles**: CSS variables, themes, and responsive breakpoints

## Theming

The ChatKit UI supports both light and dark themes with automatic detection of system preferences. The theme can be customized using CSS variables defined in `variables.css`.

## Mobile Responsiveness

The UI adapts to different screen sizes:
- Mobile: Full-screen drawer interface with slide-up animation
- Desktop: Fixed panel with launcher button
- Responsive design using CSS Grid and Flexbox
- Touch-friendly controls and interactions

## Error Handling

The system provides comprehensive error handling:
- Network error detection and recovery
- Graceful degradation when API is unavailable
- User-friendly error messages
- Retry mechanisms for failed requests
- Fallback content when streaming fails

## Future Enhancements

Potential areas for future improvement:
- Voice input/output capabilities
- File attachment support
- Conversation history persistence
- Advanced formatting options
- Multi-language support
- Enhanced accessibility features