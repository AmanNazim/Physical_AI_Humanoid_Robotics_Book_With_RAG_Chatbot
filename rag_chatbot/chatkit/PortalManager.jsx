import React, { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { ChatKitProvider } from './providers/ChatKitProvider';

const PortalManager = ({ children }) => {
  useEffect(() => {
    // Create portal root element when component mounts
    if (typeof document !== 'undefined') {
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

  // Get the portal root element after it's been created
  const portalRoot = typeof document !== 'undefined' ? document.getElementById('chatkit-portal-root') : null;

  // Only render to portal after portal root exists
  if (portalRoot) {
    return createPortal(
      <ChatKitProvider>{children}</ChatKitProvider>,
      portalRoot
    );
  }

  // Fallback: don't render until portal is ready
  return null;
};

export default PortalManager;