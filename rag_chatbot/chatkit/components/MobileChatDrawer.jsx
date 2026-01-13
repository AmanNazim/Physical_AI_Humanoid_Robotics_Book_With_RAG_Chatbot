import React, { useEffect, useRef } from 'react';
import { useChatUI } from '../contexts/ChatUIContext';
import { useChatConversation } from '../contexts/ChatConversationContext';
import ChatMessageList from './ChatMessageList';
import ChatInputBar from './ChatInputBar';
import './MobileChatDrawer.css';

const MobileChatDrawer = () => {
  const { isOpen, closeChat, isMobile } = useChatUI();
  const { messages } = useChatConversation();
  const drawerRef = useRef(null);

  // Close chat when pressing Escape
  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        closeChat();
      }
    };

    if (isOpen && isMobile) {
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, isMobile, closeChat]);

  if (!isOpen || !isMobile) {
    return null;
  }

  return (
    <div className="mobile-drawer-overlay">
      <div className="mobile-drawer" ref={drawerRef}>
        <div className="mobile-drawer-header">
          <button
            className="back-button"
            onClick={closeChat}
            aria-label="Back to content"
          >
            ←
          </button>
          <h3>AI Assistant</h3>
          <div className="header-spacer"></div>
        </div>
        <div className="mobile-drawer-content">
          <ChatMessageList messages={messages} />
          <ChatInputBar />
        </div>
      </div>
    </div>
  );
};

export default MobileChatDrawer;