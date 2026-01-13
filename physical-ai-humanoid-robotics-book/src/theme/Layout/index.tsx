import React from 'react';
import Layout from '@theme-original/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import type { Props } from '@theme/Layout';
import PortalManager from '../../../../rag_chatbot/chatkit/PortalManager';

// Simple browser-only component to handle ChatKit initialization
function ChatKitBrowserComponent() {
  React.useEffect(() => {
    // Dynamically load styles when in browser
    import('../../../../rag_chatbot/chatkit/styles/variables.css').catch(() => {});
    import('../../../../rag_chatbot/chatkit/styles/theme.css').catch(() => {});
    import('../../../../rag_chatbot/chatkit/styles/breakpoints.css').catch(() => {});
    import('../../../../rag_chatbot/chatkit/styles/animations.css').catch(() => {});
  }, []);

  return <PortalManager />;
}

export default function CustomLayout(props: Props): JSX.Element {
  return (
    <Layout {...props}>
      {props.children}
      <BrowserOnly>
        {() => <ChatKitBrowserComponent />}
      </BrowserOnly>
    </Layout>
  );
}