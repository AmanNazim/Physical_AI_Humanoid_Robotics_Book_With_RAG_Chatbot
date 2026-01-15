import { handleStream } from './streamingHandlers';
import { BACKEND_URL } from '../config/api';

/**
 * Send a message to the chat API
 * @param {Object} messageData - The message data to send
 * @param {string} messageData.message - The user message
 * @param {Object} messageData.context - Additional context (selected text, page info, etc.)
 * @param {string} messageData.sessionId - The session ID
 * @param {Function} onTokenReceived - Callback function to handle streaming tokens
 * @returns {Promise<Object>} The response from the API
 */
export const sendMessage = async (messageData, onTokenReceived) => {
  console.log('Sending message to backend:', {
    backendUrl: BACKEND_URL,
    message: messageData.message,
    sessionId: messageData.sessionId
  });

  try {
    // If streaming is requested, use the stream endpoint
    if (onTokenReceived) {
      console.log('Using streaming endpoint...');
      const response = await fetch(`${BACKEND_URL}/api/v1/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: messageData.message,
          session_id: messageData.sessionId,
          max_context: 5
        }),
      });

      console.log('Stream response received:', response.status);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle streaming response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log('Stream reading completed');
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6)); // Remove 'data: ' prefix
              if (data.type === 'token' && onTokenReceived) {
                console.log('Token received:', data.content);
                onTokenReceived(data.content);
              } else if (data.type === 'sources') {
                console.log('Sources received:', data.sources);
              } else if (data.type === 'complete') {
                console.log('Stream complete event received');
                break;
              }
            } catch (e) {
              // Skip malformed JSON
              console.warn('Malformed JSON in stream:', line);
            }
          }
        }
      }

      return { success: true };
    } else {
      // For non-streaming, use the regular chat endpoint
      console.log('Using regular chat endpoint...');
      const response = await fetch(`${BACKEND_URL}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: messageData.message,
          session_id: messageData.sessionId,
          max_context: 5
        }),
      });

      console.log('Regular response received:', response.status);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const responseData = await response.json();
      console.log('Response data received:', responseData);

      return responseData;
    }
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
};

/**
 * Send selected text to the RAG API
 * @param {Object} selectionData - The selection data to send
 * @param {string} selectionData.selectedText - The selected text
 * @param {string} selectionData.page - The current page/chapter
 * @param {string} selectionData.sessionId - The session ID
 * @returns {Promise<Object>} The response from the API
 */
export const sendSelectedText = async (selectionData) => {
  try {
    // Send selected text as a query to the regular chat endpoint
    const response = await fetch(`${BACKEND_URL}/api/v1/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: `Based on this selected text: "${selectionData.selectedText}", please provide relevant information or answer questions about it.`,
        session_id: selectionData.sessionId,
        max_context: 5
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response;
  } catch (error) {
    console.error('Error sending selected text:', error);
    throw error;
  }
};

/**
 * Get configuration from the backend
 * @returns {Promise<Object>} The configuration data
 */
export const getConfig = async () => {
  try {
    const response = await fetch(`${BACKEND_URL}/api/v1/config/`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching config:', error);
    throw error;
  }
};

/**
 * Check the health of the backend
 * @returns {Promise<boolean>} Whether the backend is healthy
 */
export const checkHealth = async () => {
  try {
    const response = await fetch(`${BACKEND_URL}/api/v1/health`);

    return response.ok;
  } catch (error) {
    console.error('Error checking health:', error);
    return false;
  }
};

export default {
  sendMessage,
  sendSelectedText,
  getConfig,
  checkHealth
};