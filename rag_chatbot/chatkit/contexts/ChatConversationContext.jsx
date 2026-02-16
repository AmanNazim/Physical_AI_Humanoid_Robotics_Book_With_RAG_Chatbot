import React, { createContext, useContext, useState, useCallback } from 'react';

const ChatConversationContext = createContext(null);

export const ChatConversationProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedText, setSelectedText] = useState(null);
  const [sessionId, setSessionId] = useState(() => {
    // Retrieve or generate session ID only in browser environment
    if (typeof window !== 'undefined') {
      return localStorage.getItem('chatkit-session-id') || `session-${Date.now()}`;
    }
    return `session-${Date.now()}`;
  });

  const addMessage = useCallback((message) => {
    setMessages(prev => [...prev, message]);
  }, []);

  const updateMessage = useCallback((id, updates) => {
    setMessages(prev => {
      const newMessages = [...prev]; // Create a new array
      const msgIndex = newMessages.findIndex(msg => msg.id === id);
      if (msgIndex !== -1) {
        // Create a completely new message object to ensure React detects the change
        newMessages[msgIndex] = { ...newMessages[msgIndex], ...updates };
      }
      return newMessages; // Return the new array to ensure React re-renders
    });
  }, []);

  // Enhanced updateMessage to support updater functions (like setState)
  const updateMessageWithUpdater = useCallback((id, updater) => {
    setMessages(prev => {
      const newMessages = [...prev]; // Create a new array
      const msgIndex = newMessages.findIndex(msg => msg.id === id);
      if (msgIndex !== -1) {
        // Call the updater function with the current message and merge the result
        const currentMessage = newMessages[msgIndex];
        const updates = typeof updater === 'function' ? updater(currentMessage) : updater;
        newMessages[msgIndex] = { ...currentMessage, ...updates };
      }
      return newMessages; // Return the new array to ensure React re-renders
    });
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
    updateMessageWithUpdater,
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