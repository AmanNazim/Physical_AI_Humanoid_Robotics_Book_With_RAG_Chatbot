import React, { useState, useEffect } from 'react';
import Layout from '@theme-original/Layout';
import type { Props } from '@theme/Layout';

// Dynamically import ChatKit components
async function loadChatKitComponents() {
  const [{ ChatKitProvider }] = await Promise.all([
    import('../../../../rag_chatbot/chatkit/providers/ChatKitProvider'),
  ]);

  // Load styles dynamically
  await Promise.all([
    import('../../../../rag_chatbot/chatkit/styles/variables.css').catch(() => {}),
    import('../../../../rag_chatbot/chatkit/styles/theme.css').catch(() => {}),
    import('../../../../rag_chatbot/chatkit/styles/breakpoints.css').catch(() => {}),
    import('../../../../rag_chatbot/chatkit/styles/animations.css').catch(() => {})
  ]);

  return { ChatKitProvider };
}

export default function CustomLayout(props: Props): JSX.Element {
  const [ChatKitComponents, setChatKitComponents] = useState<{
    ChatKitProvider: React.ComponentType<{ children: React.ReactNode }>
  } | null>(null);
  const [hasMounted, setHasMounted] = useState(false);

  useEffect(() => {
    setHasMounted(true);
  }, []);

  useEffect(() => {
    if (hasMounted) {
      loadChatKitComponents().then(setChatKitComponents);
    }
  }, [hasMounted]);

  useEffect(() => {
    // Create portal root element when ChatKit is loaded
    if (hasMounted && ChatKitComponents) {
      let portalRoot = document.getElementById('chatkit-portal-root');

      if (!portalRoot) {
        portalRoot = document.createElement('div');
        portalRoot.setAttribute('id', 'chatkit-portal-root');
        portalRoot.style.all = 'initial'; // Reset CSS inheritance
        document.body.appendChild(portalRoot);
      }

      // Cleanup function
      return () => {
        const existingPortalRoot = document.getElementById('chatkit-portal-root');
        if (existingPortalRoot && existingPortalRoot.parentNode) {
          existingPortalRoot.parentNode.removeChild(existingPortalRoot);
        }
      };
    }
  }, [hasMounted, ChatKitComponents]);

  if (ChatKitComponents && hasMounted) {
    const { ChatKitProvider } = ChatKitComponents;
    return (
      <Layout {...props}>
        <ChatKitProvider>
          {props.children}
        </ChatKitProvider>
      </Layout>
    );
  }

  // Fallback during SSR or loading
  return <Layout {...props}>{props.children}</Layout>;
}