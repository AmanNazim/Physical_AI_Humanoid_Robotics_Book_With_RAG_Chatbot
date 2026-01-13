import React, { useState, useRef, useCallback } from 'react';
import { useChatConversation } from '../contexts/ChatConversationContext';
import chatService from '../services/chatService';
import { processSSE } from '../services/streamingHandlers';
import './ChatInputBar.css';

const ChatInputBar = () => {
  const [inputValue, setInputValue] = useState('');
  const [isSending, setIsSending] = useState(false);
  const textareaRef = useRef(null);
  const {
    addMessage,
    selectedText,
    sessionId,
    updateMessage,
    setIsStreaming
  } = useChatConversation();

  // Adjust textarea height based on content
  const adjustTextareaHeight = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 150) + 'px';
    }
  }, []);

  // Handle input change
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
    adjustTextareaHeight();
  };

  // Handle key down (for Enter to send, Shift+Enter for new line)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isSending && inputValue.trim()) {
        handleSubmit();
      }
    }
  };

  // Submit handler
  const handleSubmit = async () => {
    if (!inputValue.trim() || isSending) return;

    setIsSending(true);

    // Add user message to UI immediately
    const userMessageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    addMessage({
      id: userMessageId,
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    });

    // Prepare the request payload
    const messageData = {
      message: inputValue,
      context: {
        selected_text: selectedText || null,
        page: typeof window !== 'undefined' ? window.location.pathname : ''
      },
      sessionId: sessionId
    };

    // Add bot message placeholder
    const botMessageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    addMessage({
      id: botMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true
    });

    // Set streaming state
    setIsStreaming(true);

    try {
      // Start streaming response using the updated service
      await chatService.sendMessage(messageData, async (token) => {
        // Update the bot message with the received token
        updateMessage(botMessageId, {
          content: prevContent => prevContent + token,
          isStreaming: true
        });
      });

      // Update the message to indicate streaming is complete
      updateMessage(botMessageId, {
        isStreaming: false
      });

      // Clear input after sending
      setInputValue('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    } catch (error) {
      console.error('Error sending message:', error);

      // Update the existing bot message with the error instead of creating a new message
      updateMessage(botMessageId, {
        content: `Error: ${error.message}`,
        isStreaming: false,
        error: true
      });
    } finally {
      setIsSending(false);
      setIsStreaming(false);
    }
  };

  // Disable input when streaming
  const isDisabled = isSending;

  return (
    <div className="chat-input-bar">
      {selectedText && (
        <div className="selected-text-indicator">
          Using selected text as context: "{selectedText.substring(0, 50)}{selectedText.length > 50 ? '...' : ''}"
        </div>
      )}
      <div className="input-container">
        <textarea
          ref={textareaRef}
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder={selectedText ? "Ask about the selected text..." : "Message AI assistant..."}
          className="chat-textarea"
          disabled={isDisabled}
          rows={1}
        />
        <button
          onClick={handleSubmit}
          disabled={isDisabled || !inputValue.trim()}
          className={`send-button ${inputValue.trim() ? 'active' : ''}`}
          aria-label="Send message"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </div>
    </div>
  );
};

export default ChatInputBar;