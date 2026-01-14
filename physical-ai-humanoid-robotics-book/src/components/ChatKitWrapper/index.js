import React, { useEffect } from 'react';
import { ChatKitProvider, ChatLauncherButton, ChatPanel, MobileChatDrawer } from '../../../../rag_chatbot/chatkit';

// Ensure styles are loaded
import '../../../../rag_chatbot/chatkit/styles/variables.css';
import '../../../../rag_chatbot/chatkit/styles/theme.css';
import '../../../../rag_chatbot/chatkit/styles/breakpoints.css';
import '../../../../rag_chatbot/chatkit/styles/animations.css';

const ChatKitWrapper = () => {
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
    }
  }, []);

  return (
    <ChatKitProvider>
      <ChatLauncherButton />
      <ChatPanel />
      <MobileChatDrawer />
    </ChatKitProvider>
  );
};

export default ChatKitWrapper;