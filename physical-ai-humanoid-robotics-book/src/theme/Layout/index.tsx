import React from 'react';
import Layout from '@theme-original/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import type { Props } from '@theme/Layout';

// Client-only component that handles ChatKit initialization
function ChatKitInitializer() {
  // Dynamically import and initialize ChatKit components
  React.useEffect(() => {
    // Dynamically load ChatKit styles
    import('../../../../rag_chatbot/chatkit/styles/variables.css').catch(console.warn);
    import('../../../../rag_chatbot/chatkit/styles/theme.css').catch(console.warn);
    import('../../../../rag_chatbot/chatkit/styles/breakpoints.css').catch(console.warn);
    import('../../../../rag_chatbot/chatkit/styles/animations.css').catch(console.warn);

    // Create portal root element
    if (typeof document !== 'undefined') {
      let portalRoot = document.getElementById('chatkit-portal-root');

      if (!portalRoot) {
        portalRoot = document.createElement('div');
        portalRoot.setAttribute('id', 'chatkit-portal-root');
        portalRoot.style.all = 'initial'; // Reset CSS inheritance
        document.body.appendChild(portalRoot);
      }
    }
  }, []);

  // Dynamically import and render ChatKitProvider
  const [ChatKitProvider, setChatKitProvider] = React.useState<any>(null);

  React.useEffect(() => {
    import('../../../../rag_chatbot/chatkit/providers/ChatKitProvider').then(module => {
      setChatKitProvider(() => module.ChatKitProvider);
    });
  }, []);

  if (ChatKitProvider) {
    return <ChatKitProvider />;
  }

  return null;
}

export default function CustomLayout(props: Props): JSX.Element {
  return (
    <Layout {...props}>
      {/* Original content */}
      {props.children}

      {/* ChatKit UI - only rendered in browser */}
      <BrowserOnly>
        {() => <ChatKitInitializer />}
      </BrowserOnly>
    </Layout>
  );
}