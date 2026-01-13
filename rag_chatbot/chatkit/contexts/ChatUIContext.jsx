import React, { createContext, useContext, useState } from 'react';

const ChatUIContext = createContext(null);

export const ChatUIProvider = ({ children }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  const openChat = () => setIsOpen(true);
  const closeChat = () => setIsOpen(false);
  const toggleChat = () => setIsOpen(!isOpen);

  const value = {
    isOpen,
    openChat,
    closeChat,
    toggleChat,
    isMobile,
    setIsMobile
  };

  return (
    <ChatUIContext.Provider value={value}>
      {children}
    </ChatUIContext.Provider>
  );
};

export const useChatUI = () => {
  const context = useContext(ChatUIContext);
  if (!context) {
    throw new Error('useChatUI must be used within a ChatUIProvider');
  }
  return context;
};

// Export the context for use in hooks
export { ChatUIContext };

export default ChatUIProvider;