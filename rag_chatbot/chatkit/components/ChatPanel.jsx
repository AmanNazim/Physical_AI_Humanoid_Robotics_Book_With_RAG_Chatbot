import React, { useEffect, useRef } from 'react';
import { useChatUI } from '../contexts/ChatUIContext';
import { useChatConversation } from '../contexts/ChatConversationContext';
import ChatMessageList from './ChatMessageList';
import ChatInputBar from './ChatInputBar';
import './ChatPanel.css';

const ChatPanel = () => {
  const { isOpen, closeChat } = useChatUI();
  const { isMobile } = useChatUI();
  const { messages } = useChatConversation();
  const panelRef = useRef(null);

  // Close chat when clicking outside the panel (on the backdrop)
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (panelRef.current && !panelRef.current.contains(event.target) && !event.target.closest('.chat-launcher-button')) {
        closeChat();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, closeChat]);

  // Close chat when pressing Escape
  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        closeChat();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, closeChat]);

  if (!isOpen || isMobile) {
    return null;
  }

  return (
    <div className="chat-panel-backdrop">
      <div className="chat-panel" ref={panelRef}>
        <div className="chat-header">
          <h3>AI Assistant</h3>
          <button
            className="close-button"
            onClick={closeChat}
            aria-label="Close chat"
          >
            ×
          </button>
        </div>
        <div className="chat-content">
          <ChatMessageList messages={messages} />
          <ChatInputBar />
        </div>
      </div>
    </div>
  );
};

export default ChatPanel;