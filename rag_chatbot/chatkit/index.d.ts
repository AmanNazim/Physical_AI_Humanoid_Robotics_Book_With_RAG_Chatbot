// Type declarations for ChatKit UI Subsystem

// ChatKit Provider
export const ChatKitProvider: React.FC<{ children: React.ReactNode }>;
export const useChatKit: () => {
  config: any;
  isLoading: boolean;
  error: string | null;
  messages: Array<any>;
  addMessage: (message: any) => void;
  updateMessage: (id: string, updates: any) => void;
  clearMessages: () => void;
  isStreaming: boolean;
  setIsStreaming: (streaming: boolean) => void;
  selectedText: string | null;
  setSelectedText: (text: string | null) => void;
  sessionId: string;
};

// Contexts
export const ChatUIContext: React.Context<any>;
export const useChatUI: () => {
  isOpen: boolean;
  openChat: () => void;
  closeChat: () => void;
  toggleChat: () => void;
  isMobile: boolean;
  setIsMobile: (mobile: boolean) => void;
};

export const ChatConversationContext: React.Context<any>;
export const useChatConversation: () => {
  messages: Array<any>;
  addMessage: (message: any) => void;
  updateMessage: (id: string, updates: any) => void;
  clearMessages: () => void;
  isStreaming: boolean;
  selectedText: string | null;
  setSelectedText: (text: string | null) => void;
};

// Hooks
export const useChatUI: () => any;
export const useChatMessages: () => any;
export const useStream: () => any;

// Components
export const ChatLauncherButton: React.FC;
export const ChatPanel: React.FC;
export const MobileChatDrawer: React.FC;
export const ChatMessageList: React.FC<{ messages: Array<any> }>;
export const ChatMessageBubble: React.FC<{ message: any }>;
export const MarkdownRenderer: React.FC<{ content: string }>;
export const ChatInputBar: React.FC;
export const SelectionTooltip: React.FC;
export const ErrorBubble: React.FC<{ message: string; onRetry?: () => void; showRetry?: boolean }>;

// Services
export const chatService: any;
export const processStreamEvent: (event: any) => any;
export const handleStreamError: (error: any) => any;