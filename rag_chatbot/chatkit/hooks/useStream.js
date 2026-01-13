import { useState, useCallback } from 'react';
import { useChatKit } from '../providers/ChatKitProvider';

export const useStream = () => {
  const { updateMessage, setIsStreaming } = useChatKit();
  const [activeStream, setActiveStream] = useState(null);

  const startStream = useCallback(async (messageId, streamUrl, requestBody) => {
    setIsStreaming(true);

    try {
      const response = await fetch(streamUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`Stream request failed: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Readable stream not supported');
      }

      // Process the stream
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
            if (data === '[DONE]') {
              break;
            }

            try {
              const parsed = JSON.parse(data);

              // Update the message with new content
              if (parsed.token) {
                updateMessage(messageId, {
                  content: prevContent => (prevContent || '') + parsed.token
                });
              }

              // Handle other types of stream events
              if (parsed.type === 'sources' && parsed.data) {
                updateMessage(messageId, {
                  sources: parsed.data
                });
              }

              if (parsed.type === 'error' && parsed.message) {
                updateMessage(messageId, {
                  error: parsed.message
                });
              }
            } catch (e) {
              console.warn('Could not parse stream data:', line);
            }
          }
        }
      }

    } catch (error) {
      console.error('Stream error:', error);
      updateMessage(messageId, {
        error: error.message
      });
    } finally {
      setIsStreaming(false);
      setActiveStream(null);
    }
  }, [updateMessage, setIsStreaming]);

  const cancelStream = useCallback(() => {
    if (activeStream) {
      activeStream.cancel();
      setActiveStream(null);
      setIsStreaming(false);
    }
  }, [activeStream, setIsStreaming]);

  return {
    startStream,
    cancelStream,
    isStreaming: useChatKit().isStreaming
  };
};

export default useStream;