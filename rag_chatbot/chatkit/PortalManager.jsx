import React, { useEffect } from 'react';
import { ChatKitProvider } from './providers/ChatKitProvider';

const PortalManager = ({ children }) => {
  useEffect(() => {
    // Create portal root element when component mounts
    if (typeof document !== 'undefined') {
      // Check if portal root already exists to avoid duplicates
      let portalRoot = document.getElementById('chatkit-portal-root');

      if (!portalRoot) {
        portalRoot = document.createElement('div');
        portalRoot.setAttribute('id', 'chatkit-portal-root');
        portalRoot.style.all = 'initial'; // Reset CSS inheritance
        document.body.appendChild(portalRoot);
      }

      // Clean up portal root when component unmounts
      return () => {
        const existingPortalRoot = document.getElementById('chatkit-portal-root');
        if (existingPortalRoot && existingPortalRoot.parentNode) {
          existingPortalRoot.parentNode.removeChild(existingPortalRoot);
        }
      };
    }
  }, []);

  return <ChatKitProvider>{children}</ChatKitProvider>;
};

export default PortalManager;