import React from 'react';
import { useChatUI } from '../contexts/ChatUIContext';
import { useChatConversation } from '../contexts/ChatConversationContext';
import './ChatLauncherButton.css';

const ChatLauncherButton = () => {
  const { isOpen, toggleChat } = useChatUI();
  const { isStreaming } = useChatConversation();

  return (
    <button
      className={`chat-launcher-button ${isOpen ? 'open' : ''} ${isStreaming ? 'streaming' : ''}`}
      onClick={toggleChat}
      aria-label={isOpen ? 'Close chat' : 'Open chat'}
      title={isOpen ? 'Close chat' : 'Open chat AI assistant'}
      disabled={isStreaming}
    >
      {isStreaming ? (
        <div className="loading-spinner">
          <div className="spinner-dot"></div>
          <div className="spinner-dot"></div>
          <div className="spinner-dot"></div>
        </div>
      ) : (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="chat-icon"
        >
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
      )}
    </button>
  );
};

export default ChatLauncherButton;