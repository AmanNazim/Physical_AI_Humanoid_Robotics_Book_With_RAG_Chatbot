import React, { useEffect } from 'react';
import { ChatKitProvider } from './providers/ChatKitProvider';

// PortalRootManager creates the portal root element in the DOM
const PortalRootManager = ({ children }) => {
  useEffect(() => {
    // Create portal root element when component mounts
    if (typeof document !== 'undefined') {
      const portalRoot = document.createElement('div');
      portalRoot.setAttribute('id', 'chatkit-portal-root');
      portalRoot.style.all = 'initial'; // Reset CSS inheritance
      document.body.appendChild(portalRoot);

      // Clean up portal root when component unmounts
      return () => {
        document.body.removeChild(portalRoot);
      };
    }
  }, []);

  return children;
};

const SafeChatKitProvider = ({ children }) => {
  return (
    <PortalRootManager>
      <ChatKitProvider>
        {children}
      </ChatKitProvider>
    </PortalRootManager>
  );
};

export default SafeChatKitProvider;