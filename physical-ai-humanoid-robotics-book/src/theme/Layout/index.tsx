import React, { useState, useEffect } from 'react';
import Layout from '@theme-original/Layout';
import { ChatKitProvider } from '../../../../rag_chatbot/chatkit/providers/ChatKitProvider';
import type { Props } from '@theme/Layout';

export default function CustomLayout(props: Props): JSX.Element {
  const [hasMounted, setHasMounted] = useState(false);

  useEffect(() => {
    setHasMounted(true);
  }, []);

  // Render the layout with ChatKitProvider, but ensure it only initializes in the browser
  return (
    <Layout {...props}>
      {hasMounted ? (
        <ChatKitProvider>
          {props.children}
        </ChatKitProvider>
      ) : (
        props.children
      )}
    </Layout>
  );
}