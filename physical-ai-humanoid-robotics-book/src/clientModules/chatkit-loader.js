// Client module to load ChatKit UI after page initialization
// This approach is more compatible with Docusaurus than modifying Layout

if (typeof window !== 'undefined') {
  // Wait for React to be available before proceeding
  const waitForReactAndLoadChatKit = () => {
    if (window.React && window.ReactDOM) {
      loadAndRenderChatKit();
    } else {
      setTimeout(waitForReactAndLoadChatKit, 100);
    }
  };

  const loadAndRenderChatKit = async () => {
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
        chatkitContainer.style.display = 'contents'; // Don't affect layout
        document.body.appendChild(chatkitContainer);
      }

      // Render the ChatKit UI components using React
      const element = window.React.createElement(
        PortalManager,
        null,
        window.React.createElement(ChatLauncherButton),
        window.React.createElement(ChatPanel),
        window.React.createElement(MobileChatDrawer)
      );

      const root = window.ReactDOM.createRoot(chatkitContainer);
      root.render(element);

      console.log('ChatKit UI rendered successfully');
    } catch (error) {
      console.error('Failed to load or render ChatKit UI:', error);
    }
  };

  // Start the process
  waitForReactAndLoadChatKit();
}