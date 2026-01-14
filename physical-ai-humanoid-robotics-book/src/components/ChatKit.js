import React from 'react';
import { createRoot } from 'react-dom/client';
import { PortalManager, ChatLauncherButton, ChatPanel, MobileChatDrawer } from '../../../rag_chatbot/chatkit';

// Import styles
import '../../../rag_chatbot/chatkit/styles/variables.css';
import '../../../rag_chatbot/chatkit/styles/theme.css';
import '../../../rag_chatbot/chatkit/styles/breakpoints.css';
import '../../../rag_chatbot/chatkit/styles/animations.css';

// Create portal root and render ChatKit components
if (typeof document !== 'undefined') {
  // Create portal root if it doesn't exist
  let portalRoot = document.getElementById('chatkit-portal-root');
  if (!portalRoot) {
    portalRoot = document.createElement('div');
    portalRoot.setAttribute('id', 'chatkit-portal-root');
    portalRoot.style.all = 'initial';
    document.body.appendChild(portalRoot);
  }

  // Create a container for the ChatKit UI
  const chatKitContainer = document.createElement('div');
  chatKitContainer.id = 'chatkit-app-container';
  document.body.appendChild(chatKitContainer);

  // Render the ChatKit UI
  const root = createRoot(chatKitContainer);
  root.render(
    <PortalManager>
      <ChatLauncherButton />
      <ChatPanel />
      <MobileChatDrawer />
    </PortalManager>
  );
}

export default function ChatKit() {
  // This component doesn't render anything itself
  // It's just used to initialize the ChatKit UI
  return null;
}