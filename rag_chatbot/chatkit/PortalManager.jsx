import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { ChatKitProvider } from './providers/ChatKitProvider';

const PortalManager = ({ children }) => {
  const [portalRoot, setPortalRoot] = useState(null);

  useEffect(() => {
    if (typeof document !== 'undefined') {
      // Check if portal root already exists to avoid duplicates
      let root = document.getElementById('chatkit-portal-root');

      if (!root) {
        root = document.createElement('div');
        root.setAttribute('id', 'chatkit-portal-root');
        root.style.all = 'initial'; // Reset CSS inheritance
        document.body.appendChild(root);
      }

      setPortalRoot(root);

      // Clean up portal root when component unmounts
      return () => {
        const existingRoot = document.getElementById('chatkit-portal-root');
        if (existingRoot && existingRoot.parentNode) {
          existingRoot.parentNode.removeChild(existingRoot);
        }
      };
    }
  }, []);

  // Render children using createPortal if portal root exists
  if (portalRoot) {
    return createPortal(
      <ChatKitProvider>{children}</ChatKitProvider>,
      portalRoot
    );
  }

  // Fallback: render in normal React tree if portal root doesn't exist
  return <ChatKitProvider>{children}</ChatKitProvider>;
};

export default PortalManager;