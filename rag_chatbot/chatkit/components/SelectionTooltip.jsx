import React, { useState, useEffect, useRef } from 'react';
import { useChatUI } from '../contexts/ChatUIContext';
import { useChatConversation } from '../contexts/ChatConversationContext';
import './SelectionTooltip.css';

const SelectionTooltip = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const [selectedText, setSelectedText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const tooltipRef = useRef(null);
  const { openChat } = useChatUI();
  const { setSelectedText: setContextSelectedText } = useChatConversation();

  // Handle text selection
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      const text = selection.toString().trim();

      if (text.length > 0 && text.length < 1000) { // Limit selection length
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();

        // Position the tooltip above the selection
        setPosition({
          top: rect.top + window.scrollY - 40, // 40px above selection
          left: rect.left + window.scrollX + (rect.width / 2) // Centered horizontally
        });

        setSelectedText(text);
        setIsVisible(true);
      } else {
        setIsVisible(false);
      }
    };

    const handleMouseUp = () => {
      setTimeout(handleSelection, 0); // Delay to ensure selection is complete
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('keyup', (e) => {
      if (e.key === 'Escape') {
        setIsVisible(false);
      }
    });

    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('keyup', (e) => {
        if (e.key === 'Escape') {
          setIsVisible(false);
        }
      });
    };
  }, []);

  // Handle click outside tooltip
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (tooltipRef.current && !tooltipRef.current.contains(event.target)) {
        setIsVisible(false);
      }
    };

    if (isVisible) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isVisible]);

  const handleAskAI = async () => {
    if (!selectedText || isProcessing) return;

    setIsProcessing(true);

    try {
      // Set the selected text in the context
      setContextSelectedText(selectedText);

      // Open the chat panel
      openChat();

      // Clear the selection
      window.getSelection().removeAllRanges();
    } catch (error) {
      console.error('Error handling selected text:', error);
    } finally {
      setIsProcessing(false);
      setIsVisible(false);
    }
  };

  // Don't render if not visible
  if (!isVisible || !selectedText) {
    return null;
  }

  return (
    <div
      className={`selection-tooltip ${isProcessing ? 'processing' : ''}`}
      style={{
        top: `${position.top}px`,
        left: `${position.left}px`,
        transform: 'translateX(-50%)'
      }}
      ref={tooltipRef}
    >
      <button
        className="ask-ai-button"
        onClick={handleAskAI}
        disabled={isProcessing}
        aria-label="Ask AI about selected text"
      >
        {isProcessing ? (
          <span>Asking AI...</span>
        ) : (
          <span>Ask AI about this</span>
        )}
      </button>
    </div>
  );
};

export default SelectionTooltip;