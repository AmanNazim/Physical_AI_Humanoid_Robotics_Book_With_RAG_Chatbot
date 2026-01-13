import React, { useEffect, useState } from 'react';
import Layout from '@theme-original/Layout';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import { ChatKitProvider } from '../../../../rag_chatbot/chatkit/providers/ChatKitProvider';
import type { Props } from '@theme/Layout';

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

function ChatKitWrapper({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Dynamically import and inject CSS only in the browser
    import('../../../../rag_chatbot/chatkit/styles/variables.css').catch(console.warn);
    import('../../../../rag_chatbot/chatkit/styles/theme.css').catch(console.warn);
    import('../../../../rag_chatbot/chatkit/styles/breakpoints.css').catch(console.warn);
    import('../../../../rag_chatbot/chatkit/styles/animations.css').catch(console.warn);

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

export default function CustomLayout(props: Props): JSX.Element {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    // Set client state to true after component mounts
    setIsClient(true);
  }, []);

  return (
    <Layout {...props}>
      {ExecutionEnvironment.canUseDOM && isClient ? (
        <ChatKitWrapper>{props.children}</ChatKitWrapper>
      ) : (
        props.children
      )}
    </Layout>
  );
}