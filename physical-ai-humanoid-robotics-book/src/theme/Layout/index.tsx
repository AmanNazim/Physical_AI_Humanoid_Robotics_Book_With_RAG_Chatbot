import React from 'react';
import Layout from '@theme-original/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import type { Props } from '@theme/Layout';

// Browser-only component to handle ChatKit initialization
function ChatKitBrowserComponent() {
  React.useEffect(() => {
    // Dynamically load styles when in browser
    import('../../../../rag_chatbot/chatkit/styles/variables.css').catch(() => {});
    import('../../../../rag_chatbot/chatkit/styles/theme.css').catch(() => {});
    import('../../../../rag_chatbot/chatkit/styles/breakpoints.css').catch(() => {});
    import('../../../../rag_chatbot/chatkit/styles/animations.css').catch(() => {});

    // Create portal root element when in browser
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

  // Dynamically import and render ChatKitProvider
  const [ChatKitProvider, setChatKitProvider] = React.useState<any>(null);

  React.useEffect(() => {
    import('../../../../rag_chatbot/chatkit/providers/ChatKitProvider')
      .then(module => setChatKitProvider(() => module.ChatKitProvider))
      .catch(err => console.error('Failed to load ChatKitProvider:', err));
  }, []);

  if (ChatKitProvider) {
    return React.createElement(ChatKitProvider);
  }
  return null;
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