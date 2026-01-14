import React, { useEffect, useState } from 'react';

// Import all necessary ChatKit components and styles
import { ChatKitProvider, ChatLauncherButton, ChatPanel, MobileChatDrawer } from '../../../../rag_chatbot/chatkit';
import '../../../../rag_chatbot/chatkit/styles/variables.css';
import '../../../../rag_chatbot/chatkit/styles/theme.css';
import '../../../../rag_chatbot/chatkit/styles/breakpoints.css';
import '../../../../rag_chatbot/chatkit/styles/animations.css';

const ChatKitUI = () => {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Create portal root element if it doesn't exist
    if (typeof document !== 'undefined') {
      let portalRoot = document.getElementById('chatkit-portal-root');
      if (!portalRoot) {
        portalRoot = document.createElement('div');
        portalRoot.setAttribute('id', 'chatkit-portal-root');
        portalRoot.style.all = 'initial';
        document.body.appendChild(portalRoot);
      }

      setIsLoaded(true);
    }
  }, []);

  if (!isLoaded) {
    return null;
  }

  return (
    <ChatKitProvider>
      <div style={{ display: 'contents' }}>
        <ChatLauncherButton />
        <ChatPanel />
        <MobileChatDrawer />
      </div>
    </ChatKitProvider>
  );
};

export default ChatKitUI;