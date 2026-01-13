// Client module to load ChatKit UI after page initialization
// This approach is more compatible with Docusaurus than modifying Layout

if (typeof window !== 'undefined') {
  // Wait for the page to fully load
  window.addEventListener('load', async () => {
    try {
      // Load ChatKit styles
      await Promise.all([
        import('../../../rag_chatbot/chatkit/styles/variables.css'),
        import('../../../rag_chatbot/chatkit/styles/theme.css'),
        import('../../../rag_chatbot/chatkit/styles/breakpoints.css'),
        import('../../../rag_chatbot/chatkit/styles/animations.css')
      ]);

      // Create portal root element if it doesn't exist
      if (typeof document !== 'undefined') {
        let portalRoot = document.getElementById('chatkit-portal-root');
        if (!portalRoot) {
          portalRoot = document.createElement('div');
          portalRoot.setAttribute('id', 'chatkit-portal-root');
          portalRoot.style.all = 'initial';
          document.body.appendChild(portalRoot);
        }
      }

      // Dynamically load and initialize ChatKit
      const chatkitModule = await import('../../../rag_chatbot/chatkit');
      const PortalManager = chatkitModule.PortalManager || chatkitModule.default?.PortalManager || chatkitModule;

      // The PortalManager will handle creating the UI components
      // We just need to ensure the portal root exists and the styles are loaded
      console.log('ChatKit loaded successfully');
    } catch (error) {
      console.error('Failed to load ChatKit UI:', error);
    }
  });
}