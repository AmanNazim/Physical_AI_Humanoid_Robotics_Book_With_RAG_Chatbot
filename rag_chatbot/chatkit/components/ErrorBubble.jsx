import React from 'react';
import './ErrorBubble.css';

const ErrorBubble = ({ message, onRetry, showRetry = true }) => {
  const handleRetry = () => {
    if (onRetry) {
      onRetry();
    }
  };

  return (
    <div className="error-bubble">
      <div className="error-content">
        <div className="error-icon">⚠️</div>
        <div className="error-message">{message}</div>
      </div>
      {showRetry && (
        <div className="error-actions">
          <button className="retry-button" onClick={handleRetry}>
            Try Again
          </button>
        </div>
      )}
    </div>
  );
};

export default ErrorBubble;