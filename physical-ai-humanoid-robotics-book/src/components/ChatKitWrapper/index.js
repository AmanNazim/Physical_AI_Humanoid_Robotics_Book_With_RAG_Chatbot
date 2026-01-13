import React, { useEffect } from 'react';
import { PortalManager, ChatLauncherButton, ChatPanel, MobileChatDrawer } from '../../../../rag_chatbot/chatkit';

// Ensure styles are loaded
import '../../../../rag_chatbot/chatkit/styles/variables.css';
import '../../../../rag_chatbot/chatkit/styles/theme.css';
import '../../../../rag_chatbot/chatkit/styles/breakpoints.css';
import '../../../../rag_chatbot/chatkit/styles/animations.css';

const ChatKitWrapper = () => {
  useEffect(() => {
    // PortalManager handles portal root creation internally
    // No additional setup needed here
  }, []);

  return (
    <PortalManager>
      <ChatLauncherButton />
      <ChatPanel />
      <MobileChatDrawer />
    </PortalManager>
  );
};

export default ChatKitWrapper;