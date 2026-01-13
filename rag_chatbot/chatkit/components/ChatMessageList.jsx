import React, { useRef, useEffect } from 'react';
import ChatMessageBubble from './ChatMessageBubble';
import './ChatMessageList.css';

const ChatMessageList = ({ messages }) => {
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Handle manual scrolling to enable/disable auto-scroll
  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
      // If user scrolls up more than 100px from bottom, disable auto-scroll
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
      if (distanceFromBottom > 100) {
        // User scrolled up, don't auto-scroll
        return;
      }
    }
  };

  return (
    <div className="chat-message-list" ref={messagesContainerRef} onScroll={handleScroll}>
      {messages.map((message) => (
        <ChatMessageBubble
          key={message.id}
          message={message}
        />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatMessageList;