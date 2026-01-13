import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import ChatUIProvider, { ChatUIContext } from '../contexts/ChatUIContext';
import ChatConversationProvider, { ChatConversationContext } from '../contexts/ChatConversationContext';
import { BACKEND_URL } from '../config/api';


// Create the context
const ChatKitContext = createContext(null);

// ChatKit Provider Component
export const ChatKitProvider = ({ children }) => {
  const [config, setConfig] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedText, setSelectedText] = useState(null);
  const [sessionId, setSessionId] = useState(() => {
    // Generate or retrieve session ID only in browser environment
    if (typeof window !== 'undefined') {
      return localStorage.getItem('chatkit-session-id') || `session-${Date.now()}`;
    }
    return `session-${Date.now()}`;
  });

    // Load configuration from backend
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/v1/config/chatkit`);
        if (!response.ok) {
          throw new Error(`Failed to load config: ${response.status}`);
        }
        const configData = await response.json();
        setConfig(configData);
      } catch (err) {
        console.error('Error loading ChatKit config:', err);
        setError(err.message);
        // Use default config if API fails
        setConfig({
          theme: 'light',
          maxTokens: 1000,
          temperature: 0.7
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadConfig();

    // Save session ID to localStorage only in browser environment
    if (typeof window !== 'undefined') {
      localStorage.setItem('chatkit-session-id', sessionId);
    }
  }, [sessionId]);


  // Function to add a new message
  const addMessage = useCallback((message) => {
    setMessages(prev => [...prev, message]);
  }, []);

  // Function to update a message (useful for streaming)
  const updateMessage = useCallback((id, updates) => {
    setMessages(prev =>
      prev.map(msg => msg.id === id ? { ...msg, ...updates } : msg)
    );
  }, []);

  // Function to clear messages
  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  // Value to be provided to consumers
  const value = {
    config,
    isLoading,
    error,
    messages,
    addMessage,
    updateMessage,
    clearMessages,
    isStreaming,
    setIsStreaming,
    selectedText,
    setSelectedText,
    sessionId
  };

  return (
    <ChatKitContext.Provider value={value}>
      <ChatUIProvider>
        <ChatConversationProvider>
          {children}
        </ChatConversationProvider>
      </ChatUIProvider>
    </ChatKitContext.Provider>
  );
};

// Custom hook to use the ChatKit context
export const useChatKit = () => {
  const context = useContext(ChatKitContext);
  if (!context) {
    throw new Error('useChatKit must be used within a ChatKitProvider');
  }
  return context;
};

export default ChatKitProvider;