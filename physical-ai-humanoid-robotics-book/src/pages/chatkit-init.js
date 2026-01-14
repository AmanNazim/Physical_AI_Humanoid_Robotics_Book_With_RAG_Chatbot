// Client module to initialize ChatKit UI
if (typeof window !== 'undefined') {
  // Function to initialize ChatKit
  async function initChatKit() {
    try {
      // Load required styles
      await Promise.all([
        import('../../rag_chatbot/chatkit/styles/variables.css'),
        import('../../rag_chatbot/chatkit/styles/theme.css'),
        import('../../rag_chatbot/chatkit/styles/breakpoints.css'),
        import('../../rag_chatbot/chatkit/styles/animations.css')
      ]);

      // Wait for React and ReactDOM to be available
      if (window.React && window.ReactDOM) {
        await loadAndRenderChatKit();
      } else {
        // Poll for React availability
        const checkReact = () => {
          if (window.React && window.ReactDOM && window.ReactDOM.createRoot) {
            loadAndRenderChatKit();
          } else {
            setTimeout(checkReact, 100);
          }
        };
        checkReact();
      }
    } catch (error) {
      console.error('Error initializing ChatKit:', error);
    }
  }

  async function loadAndRenderChatKit() {
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

      // Render the ChatKit UI
      const element = window.React.createElement(
        PortalManager,
        null,
        window.React.createElement(ChatLauncherButton),
        window.React.createElement(ChatPanel),
        window.React.createElement(MobileChatDrawer)
      );

      const root = window.ReactDOM.createRoot(chatkitContainer);
      root.render(element);
    } catch (error) {
      console.error('Error loading ChatKit components:', error);
    }
  }

  // Initialize when page is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initChatKit);
  } else {
    initChatKit();
  }
}