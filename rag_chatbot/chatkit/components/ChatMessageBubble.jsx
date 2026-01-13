import React from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import './ChatMessageBubble.css';

const ChatMessageBubble = ({ message }) => {
  const isUser = message.role === 'user';
  const isBot = message.role === 'assistant';
  const isError = message.error;
  const isStreaming = message.isStreaming;

  return (
    <div className={`message-bubble ${isUser ? 'user-message' : isBot ? 'bot-message' : ''} ${isError ? 'error-message' : ''}`}>
      <div className="message-avatar">
        {isUser ? (
          <span className="user-avatar">👤</span>
        ) : (
          <span className="bot-avatar">🤖</span>
        )}
      </div>
      <div className="message-content">
        {isError ? (
          <div className="error-content">
            <MarkdownRenderer content={message.content || message.error || 'An error occurred'} />
          </div>
        ) : (
          <>
            <MarkdownRenderer content={message.content || message.text || ''} />
            {isStreaming && (
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            )}
            {message.sources && message.sources.length > 0 && (
              <div className="message-sources">
                <details>
                  <summary>Sources</summary>
                  <ul>
                    {message.sources.map((source, index) => (
                      <li key={index}>{source.title || source.text?.substring(0, 50) + '...'}</li>
                    ))}
                  </ul>
                </details>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ChatMessageBubble;