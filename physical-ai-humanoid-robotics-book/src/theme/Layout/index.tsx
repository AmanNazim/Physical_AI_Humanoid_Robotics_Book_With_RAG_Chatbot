import React from 'react';
import Layout from '@theme-original/Layout';
import SafeChatKitProvider from '../../../../rag_chatbot/chatkit/SafeChatKitProvider';
import type { Props } from '@theme/Layout';

export default function CustomLayout(props: Props): JSX.Element {
  return (
    <Layout {...props}>
      <SafeChatKitProvider>
        {props.children}
      </SafeChatKitProvider>
    </Layout>
  );
}