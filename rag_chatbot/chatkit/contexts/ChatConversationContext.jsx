import React, { createContext, useContext, useState, useCallback } from 'react';

const ChatConversationContext = createContext(null);

export const ChatConversationProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedText, setSelectedText] = useState(null);
  const [sessionId, setSessionId] = useState(() => {
    // Retrieve or generate session ID
    return localStorage.getItem('chatkit-session-id') || `session-${Date.now()}`;
  });

  const addMessage = useCallback((message) => {
    setMessages(prev => [...prev, message]);
  }, []);

  const updateMessage = useCallback((id, updates) => {
    setMessages(prev =>
      prev.map(msg => msg.id === id ? { ...msg, ...updates } : msg)
    );
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  const setStreamingState = useCallback((streaming) => {
    setIsStreaming(streaming);
  }, []);

  // Define setIsStreaming callback for direct access
  const setIsStreamingCallback = useCallback((streaming) => {
    setIsStreaming(streaming);
  }, []);

  const value = {
    messages,
    addMessage,
    updateMessage,
    clearMessages,
    isStreaming,
    setIsStreaming: setIsStreamingCallback, // Direct setter for streaming state
    setStreamingState,
    selectedText,
    setSelectedText,
    sessionId
  };

  return (
    <ChatConversationContext.Provider value={value}>
      {children}
    </ChatConversationContext.Provider>
  );
};

export const useChatConversation = () => {
  const context = useContext(ChatConversationContext);
  if (!context) {
    throw new Error('useChatConversation must be used within a ChatConversationProvider');
  }
  return context;
};

export default ChatConversationProvider;