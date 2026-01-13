import React, { useState, useEffect } from 'react';
import Layout from '@theme-original/Layout';
import type { Props } from '@theme/Layout';

export default function CustomLayout(props: Props): JSX.Element {
  const [ChatKitInjector, setChatKitInjector] = useState<React.ComponentType<{ children: React.ReactNode }> | null>(null);

  useEffect(() => {
    // Dynamically import ChatKitInjector only on the client side
    import('../components/ChatKitInjector').then(module => {
      setChatKitInjector(() => module.default);
    });
  }, []);

  if (ChatKitInjector) {
    return (
      <Layout {...props}>
        <ChatKitInjector>{props.children}</ChatKitInjector>
      </Layout>
    );
  }

  // Fallback to regular layout without ChatKit during SSR or until component loads
  return <Layout {...props}>{props.children}</Layout>;
}