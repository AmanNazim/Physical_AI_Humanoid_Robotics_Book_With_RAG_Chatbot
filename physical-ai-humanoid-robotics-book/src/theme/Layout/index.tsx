import React, { useEffect } from 'react';
import Layout from '@theme-original/Layout';
import { ChatKitProvider } from '../../../../rag_chatbot/chatkit/providers/ChatKitProvider';
import '../../../../rag_chatbot/chatkit/styles/variables.css';
import '../../../../rag_chatbot/chatkit/styles/theme.css';
import '../../../../rag_chatbot/chatkit/styles/breakpoints.css';
import '../../../../rag_chatbot/chatkit/styles/animations.css';
import type { Props } from '@theme/Layout';

// Create portal root element for chat widget
const createPortalRoot = () => {
  const portalRoot = document.createElement('div');
  portalRoot.setAttribute('id', 'chatkit-portal-root');
  portalRoot.style.all = 'initial'; // Reset CSS inheritance
  return portalRoot;
};

export default function CustomLayout(props: Props): JSX.Element {
  useEffect(() => {
    // Create portal root element when component mounts
    const portalRoot = createPortalRoot();
    document.body.appendChild(portalRoot);

    // Clean up portal root when component unmounts
    return () => {
      document.body.removeChild(portalRoot);
    };
  }, []);

  return (
    <Layout {...props}>
      <ChatKitProvider>
        {props.children}
      </ChatKitProvider>
    </Layout>
  );
}