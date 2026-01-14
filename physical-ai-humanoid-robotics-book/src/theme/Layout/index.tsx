import React from 'react';
import Layout from '@theme-original/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import type { Props } from '@theme/Layout';
import ChatKitUI from '@site/src/components/ChatKitUI/ChatKitUI';

export default function CustomLayout(props: Props): JSX.Element {
  return (
    <Layout {...props}>
      {props.children}
      <BrowserOnly>
        <ChatKitUI />
      </BrowserOnly>
    </Layout>
  );
}