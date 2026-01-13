import React from 'react';
import Layout from '@theme-original/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import type { Props } from '@theme/Layout';
import ChatKitWrapper from '@site/src/components/ChatKitWrapper';

export default function CustomLayout(props: Props): JSX.Element {
  return (
    <Layout {...props}>
      {props.children}
      <BrowserOnly>
        <ChatKitWrapper />
      </BrowserOnly>
    </Layout>
  );
}