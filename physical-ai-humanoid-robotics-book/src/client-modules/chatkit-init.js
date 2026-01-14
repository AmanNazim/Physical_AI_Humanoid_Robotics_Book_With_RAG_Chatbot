// Client module to initialize ChatKit UI
// This file is loaded on the client-side after the page loads

if (typeof window !== 'undefined') {
  // Wait for the page and React to be ready
  const initChatKit = async () => {
    try {
      // Load ChatKit styles
      await Promise.all([
        import('../../rag_chatbot/chatkit/styles/variables.css'),
        import('../../rag_chatbot/chatkit/styles/theme.css'),
        import('../../rag_chatbot/chatkit/styles/breakpoints.css'),
        import('../../rag_chatbot/chatkit/styles/animations.css')
      ]);

      // Wait for React to be available before proceeding
      const waitForReact = () => {
        if (window.React && window.ReactDOM && window.ReactDOM.createRoot) {
          loadAndRenderChatKit();
        } else {
          setTimeout(waitForReact, 100);
        }
      };

      waitForReact();
    } catch (error) {
      console.error('Error loading ChatKit styles:', error);
    }
  };

  const loadAndRenderChatKit = async () => {
    try {
      // Load ChatKit components
      const chatkitModule = await import('../../rag_chatbot/chatkit');
      const { PortalManager, ChatLauncherButton, ChatPanel, MobileChatDrawer } = chatkitModule;

      // Create portal root if it doesn't exist
      if (typeof document !== 'undefined') {
        let portalRoot = document.getElementById('chatkit-portal-root');
        if (!portalRoot) {
          portalRoot = document.createElement('div');
          portalRoot.setAttribute('id', 'chatkit-portal-root');
          portalRoot.style.all = 'initial';
          document.body.appendChild(portalRoot);
        }
      }

      // Create container for ChatKit UI
      let chatkitContainer = document.getElementById('chatkit-ui-container');
      if (!chatkitContainer) {
        chatkitContainer = document.createElement('div');
        chatkitContainer.id = 'chatkit-ui-container';
        document.body.appendChild(chatkitContainer);
      }

      // Render the ChatKit UI components
      const element = window.React.createElement(
        PortalManager,
        null,
        window.React.createElement(ChatLauncherButton),
        window.React.createElement(ChatPanel),
        window.React.createElement(MobileChatDrawer)
      );

      const root = window.ReactDOM.createRoot(chatkitContainer);
      root.render(element);

      console.log('ChatKit UI initialized successfully');
    } catch (error) {
      console.error('Error initializing ChatKit UI:', error);
    }
  };

  // Initialize when the page is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initChatKit);
  } else {
    initChatKit();
  }
}