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

      // Dynamically load ChatKit components
      const chatkit = await import('../../../rag_chatbot/chatkit');
      const { PortalManager, ChatLauncherButton, ChatPanel, MobileChatDrawer } = chatkit;

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

      // Create a container element for the ChatKit UI components
      let chatkitContainer = document.getElementById('chatkit-ui-container');
      if (!chatkitContainer) {
        chatkitContainer = document.createElement('div');
        chatkitContainer.id = 'chatkit-ui-container';
        document.body.appendChild(chatkitContainer);
      }

      // Use React to render the ChatKit UI components
      // Wait for React and ReactDOM to be available
      if (window.React && window.ReactDOM) {
        // Direct render approach
        const element = window.React.createElement(
          PortalManager,
          null,
          window.React.createElement(ChatLauncherButton),
          window.React.createElement(ChatPanel),
          window.React.createElement(MobileChatDrawer)
        );
        const root = window.ReactDOM.createRoot(chatkitContainer);
        root.render(element);
      } else {
        // Alternative approach using setTimeout to wait for React
        const waitForReactAndRender = () => {
          if (window.React && window.ReactDOM) {
            const element = window.React.createElement(
              PortalManager,
              null,
              window.React.createElement(ChatLauncherButton),
              window.React.createElement(ChatPanel),
              window.React.createElement(MobileChatDrawer)
            );
            const root = window.ReactDOM.createRoot(chatkitContainer);
            root.render(element);
          } else {
            setTimeout(waitForReactAndRender, 100);
          }
        };
        waitForReactAndRender();
      }
    } catch (error) {
      console.error('Failed to load or render ChatKit UI:', error);
    }
  });
}