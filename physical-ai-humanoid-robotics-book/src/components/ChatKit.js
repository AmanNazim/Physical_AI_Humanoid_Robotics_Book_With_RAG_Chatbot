import React from 'react';
import { PortalManager, ChatLauncherButton, ChatPanel, MobileChatDrawer, SelectionTooltip } from '../../../rag_chatbot/chatkit';

// Import styles
import '../../../rag_chatbot/chatkit/styles/variables.css';
import '../../../rag_chatbot/chatkit/styles/theme.css';
import '../../../rag_chatbot/chatkit/styles/breakpoints.css';
import '../../../rag_chatbot/chatkit/styles/animations.css';

export default function ChatKit() {
  // This component renders the ChatKit UI using PortalManager
  // PortalManager handles creating the portal root and wrapping with providers
  return (
    <PortalManager>
      <SelectionTooltip />
      <ChatLauncherButton />
      <ChatPanel />
      <MobileChatDrawer />
    </PortalManager>
  );
}