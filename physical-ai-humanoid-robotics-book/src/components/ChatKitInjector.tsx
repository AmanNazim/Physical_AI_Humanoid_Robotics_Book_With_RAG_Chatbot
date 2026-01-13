import React, { useEffect } from 'react';
import { ChatKitProvider } from '../../../rag_chatbot/chatkit/providers/ChatKitProvider';

// Create portal root element for chat widget
const createPortalRoot = () => {
  if (typeof document !== 'undefined') {
    const portalRoot = document.createElement('div');
    portalRoot.setAttribute('id', 'chatkit-portal-root');
    portalRoot.style.all = 'initial'; // Reset CSS inheritance
    return portalRoot;
  }
  return null;
};

export default function ChatKitInjector({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Dynamically import and inject CSS only in the browser
    import('../../../rag_chatbot/chatkit/styles/variables.css').catch(console.warn);
    import('../../../rag_chatbot/chatkit/styles/theme.css').catch(console.warn);
    import('../../../rag_chatbot/chatkit/styles/breakpoints.css').catch(console.warn);
    import('../../../rag_chatbot/chatkit/styles/animations.css').catch(console.warn);

    // Create portal root element when component mounts
    const portalRoot = createPortalRoot();
    if (portalRoot && typeof document !== 'undefined') {
      document.body.appendChild(portalRoot);
    }

    // Clean up portal root when component unmounts
    return () => {
      if (portalRoot && typeof document !== 'undefined') {
        document.body.removeChild(portalRoot);
      }
    };
  }, []);

  return (
    <ChatKitProvider>
      {children}
    </ChatKitProvider>
  );
}