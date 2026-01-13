/**
 * Handle streaming responses from the API
 * @param {Response} response - The fetch response object with streaming data
 * @param {Function} onToken - Callback function to handle each token
 * @param {Function} onComplete - Callback function when stream completes
 * @param {Function} onError - Callback function when an error occurs
 */
export const handleStream = async (response, onToken, onComplete, onError) => {
  try {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep last incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6); // Remove 'data: ' prefix

          if (data === '[DONE]' || data.trim() === '') {
            continue;
          }

          try {
            const parsed = JSON.parse(data);

            // Handle different types of stream events
            if (parsed.type === 'token' && parsed.content) {
              onToken(parsed.content);
            } else if (parsed.type === 'sources' && parsed.data) {
              onToken({ type: 'sources', data: parsed.data });
            } else if (parsed.type === 'error' && parsed.message) {
              onError(parsed.message);
            } else if (parsed.type === 'complete') {
              onComplete(parsed);
            }
          } catch (e) {
            console.warn('Could not parse stream data:', line, e);
          }
        }
      }
    }

    // Process any remaining data in buffer
    if (buffer.trim()) {
      const line = buffer.trim();
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        try {
          const parsed = JSON.parse(data);
          if (parsed.type === 'token' && parsed.content) {
            onToken(parsed.content);
          }
        } catch (e) {
          console.warn('Could not parse stream data:', line, e);
        }
      }
    }

    onComplete();
  } catch (error) {
    console.error('Stream error:', error);
    onError(error.message);
  }
};

/**
 * Process Server-Sent Events from the response
 * @param {Response} response - The fetch response object
 * @param {Function} onToken - Callback function to handle each token
 * @param {Function} onComplete - Callback function when stream completes
 * @param {Function} onError - Callback function when an error occurs
 */
export const processSSE = async (response, onToken, onComplete, onError) => {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep last incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6); // Remove 'data: ' prefix

          if (data === '[DONE]' || data.trim() === '') {
            continue;
          }

          try {
            const parsed = JSON.parse(data);

            if (parsed.type === 'token' && parsed.content) {
              onToken(parsed.content);
            } else if (parsed.type === 'sources') {
              console.log('Sources received:', parsed.sources);
            } else if (parsed.type === 'complete') {
              onComplete(parsed);
            } else if (parsed.type === 'error') {
              onError(parsed.message || 'Stream error occurred');
            }
          } catch (e) {
            console.warn('Could not parse SSE data:', line, e);
          }
        }
      }
    }

    // Process any remaining data in buffer
    if (buffer.trim()) {
      const line = buffer.trim();
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        try {
          const parsed = JSON.parse(data);
          if (parsed.type === 'token' && parsed.content) {
            onToken(parsed.content);
          }
        } catch (e) {
          console.warn('Could not parse remaining SSE data:', line, e);
        }
      }
    }

    onComplete();
  } catch (error) {
    console.error('SSE processing error:', error);
    onError(error.message);
  } finally {
    reader.releaseLock();
  }
};

export default {
  handleStream,
  processSSE
};