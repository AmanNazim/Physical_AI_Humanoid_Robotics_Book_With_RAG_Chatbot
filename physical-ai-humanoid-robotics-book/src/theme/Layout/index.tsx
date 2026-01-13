import React from 'react';
import Layout from '@theme-original/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import type { Props } from '@theme/Layout';

// Browser-only component to handle ChatKit UI
function ChatKitBrowserComponent() {
  const [ChatKitComponents, setChatKitComponents] = React.useState<any>(null);

  React.useEffect(() => {
    // Dynamically load styles when in browser
    import('../../../../rag_chatbot/chatkit/styles/variables.css').catch(() => {});
    import('../../../../rag_chatbot/chatkit/styles/theme.css').catch(() => {});
    import('../../../../rag_chatbot/chatkit/styles/breakpoints.css').catch(() => {});
    import('../../../../rag_chatbot/chatkit/styles/animations.css').catch(() => {});

    // Load ChatKit components
    import('../../../../rag_chatbot/chatkit')
      .then(module => {
        setChatKitComponents(module);
      })
      .catch(err => console.error('Failed to load ChatKit components:', err));
  }, []);

  if (ChatKitComponents) {
    const { PortalManager, ChatLauncherButton, ChatPanel, MobileChatDrawer } = ChatKitComponents;

    // Render the complete ChatKit UI within the PortalManager
    return React.createElement(PortalManager, {},
      React.createElement(() => (
        <>
          {React.createElement(ChatLauncherButton)}
          {React.createElement(ChatPanel)}
          {React.createElement(MobileChatDrawer)}
        </>
      ), {})
    );
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