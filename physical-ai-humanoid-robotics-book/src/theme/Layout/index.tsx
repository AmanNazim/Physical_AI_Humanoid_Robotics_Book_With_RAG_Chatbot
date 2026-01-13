import React from 'react';
import Layout from '@theme-original/Layout';
import { ChatKitProvider } from '../../../../rag_chatbot/chatkit/providers/ChatKitProvider';
import type { Props } from '@theme/Layout';

export default function CustomLayout(props: Props): JSX.Element {
  return (
    <Layout {...props}>
      <ChatKitProvider>
        {props.children}
      </ChatKitProvider>
    </Layout>
  );
}