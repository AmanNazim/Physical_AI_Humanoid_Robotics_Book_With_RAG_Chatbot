import React, { useEffect } from 'react';
import Layout from '@theme-original/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { ChatKitProvider } from '../../../../rag_chatbot/chatkit/providers/ChatKitProvider';
import type { Props } from '@theme/Layout';

// Create portal root element for chat widget
const createPortalRoot = () => {
  const portalRoot = document.createElement('div');
  portalRoot.setAttribute('id', 'chatkit-portal-root');
  portalRoot.style.all = 'initial'; // Reset CSS inheritance
  return portalRoot;
};

function ChatKitWrapper({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Dynamically import and inject CSS only in the browser
    import('../../../../rag_chatbot/chatkit/styles/variables.css');
    import('../../../../rag_chatbot/chatkit/styles/theme.css');
    import('../../../../rag_chatbot/chatkit/styles/breakpoints.css');
    import('../../../../rag_chatbot/chatkit/styles/animations.css');

    // Create portal root element when component mounts
    const portalRoot = createPortalRoot();
    document.body.appendChild(portalRoot);

    // Clean up portal root when component unmounts
    return () => {
      document.body.removeChild(portalRoot);
    };
  }, []);

  return (
    <ChatKitProvider>
      {children}
    </ChatKitProvider>
  );
}

export default function CustomLayout(props: Props): JSX.Element {
  return (
    <Layout {...props}>
      <BrowserOnly>
        {() => <ChatKitWrapper>{props.children}</ChatKitWrapper>}
      </BrowserOnly>
    </Layout>
  );
}