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

      if (text.length > 0 && text.length < 2000 && selection.rangeCount > 0) { // Limit selection length
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();

        // The tooltip CSS is position: absolute, so use DOCUMENT coordinates
        // (viewport rect + scroll offset). This anchors the pill to the
        // selected text: it scrolls WITH the page instead of sticking in
        // the viewport. Clamped so it was fully visible where the selection
        // was made.
        setPosition({
          top: Math.max(window.scrollY + 8, rect.top + window.scrollY - 44), // 44px above selection
          left: Math.min(
            Math.max(rect.left + window.scrollX + (rect.width / 2), window.scrollX + 90), // centered on selection,
            window.scrollX + window.innerWidth - 90                                    // but never past the edges
          )
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

    const handleKeyUp = (e) => {
      if (e.key === 'Escape') {
        setIsVisible(false);
      }
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('keyup', handleKeyUp);

    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('keyup', handleKeyUp);
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