import React, { useEffect } from 'react';
import Layout from '@theme-original/Layout';
import { ChatKitProvider } from '../../../../rag_chatbot/chatkit/providers/ChatKitProvider';
import type { Props } from '@theme/Layout';

export default function CustomLayout(props: Props): JSX.Element {
  // This component will render both during SSR and in the browser
  // The ChatKitProvider handles browser-only operations internally via useEffect
  return (
    <Layout {...props}>
      <ChatKitProvider>
        {props.children}
      </ChatKitProvider>
    </Layout>
  );
}