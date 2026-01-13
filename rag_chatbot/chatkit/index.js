// ChatKit UI Subsystem - Main Export
// This file provides a clean import interface for the ChatKit components

// Provider
export { ChatKitProvider, useChatKit } from './providers/ChatKitProvider';
export { default as SafeChatKitProvider } from './SafeChatKitProvider';
export { default as PortalManager } from './PortalManager';

// Contexts
export { ChatUIContext, useChatUI } from './contexts/ChatUIContext';
export { ChatConversationContext, useChatConversation } from './contexts/ChatConversationContext';

// Hooks
export { useChatMessages } from './hooks/useChatMessages';
export { useStream } from './hooks/useStream';

// Components
export { default as ChatLauncherButton } from './components/ChatLauncherButton';
export { default as ChatPanel } from './components/ChatPanel';
export { default as MobileChatDrawer } from './components/MobileChatDrawer';
export { default as ChatMessageList } from './components/ChatMessageList';
export { default as ChatMessageBubble } from './components/ChatMessageBubble';
export { default as MarkdownRenderer } from './components/MarkdownRenderer';
export { default as ChatInputBar } from './components/ChatInputBar';
export { default as SelectionTooltip } from './components/SelectionTooltip';
export { default as ErrorBubble } from './components/ErrorBubble';

// Services
export { default as chatService } from './services/chatService';
export { processStreamEvent, handleStreamError } from './services/streamingHandlers';

// Styles
// Import these in your application to apply ChatKit styles
// import './styles/variables.css';
// import './styles/theme.css';
// import './styles/breakpoints.css';
// import './styles/animations.css';