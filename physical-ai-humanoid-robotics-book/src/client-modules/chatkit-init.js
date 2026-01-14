// Client module to initialize ChatKit UI
// This file is loaded on the client-side after the page loads

if (typeof window !== 'undefined') {
  // Wait for the page and React to be ready
  const initChatKit = async () => {
    try {
      // Load ChatKit styles - correct relative paths from the book directory
      await Promise.all([
        import('@site/../rag_chatbot/chatkit/styles/variables.css'),
        import('@site/../rag_chatbot/chatkit/styles/theme.css'),
        import('@site/../rag_chatbot/chatkit/styles/breakpoints.css'),
        import('@site/../rag_chatbot/chatkit/styles/animations.css')
      ]);

      // Wait for React to be available before proceeding
      // In Docusaurus, React might be available under different property names
      const waitForReact = () => {
        // Check multiple possible locations for React in Docusaurus
        const reactLib = window.React || window.Docusaurus?.React || window.require?.('react');
        const reactDOMLib = window.ReactDOM || window.Docusaurus?.ReactDOM || window.require?.('react-dom/client');

        if (reactLib && reactDOMLib && reactDOMLib.createRoot) {
          loadAndRenderChatKit(reactLib, reactDOMLib);
        } else {
          // Try again after a short delay
          setTimeout(waitForReact, 100);
        }
      };

      waitForReact();
    } catch (error) {
      console.error('Error loading ChatKit styles:', error);
    }
  };

  const loadAndRenderChatKit = async (React, ReactDOM) => {
    try {
      // Load ChatKit components - correct relative path
      const chatkitModule = await import('@site/../rag_chatbot/chatkit');
      const { PortalManager, ChatLauncherButton, ChatPanel, MobileChatDrawer } = chatkitModule;

      // Create a simple div to render the PortalManager into
      // The PortalManager will handle creating the portal root and rendering children to it
      let chatkitApp = document.getElementById('chatkit-app');
      if (!chatkitApp) {
        chatkitApp = document.createElement('div');
        chatkitApp.id = 'chatkit-app';
        chatkitApp.style.display = 'none'; // Hide this since the actual UI will be in the portal
        document.body.appendChild(chatkitApp);
      }

      // Render the PortalManager with its children
      // PortalManager will create the portal root and render the children there
      const element = React.createElement(
        PortalManager,
        null,
        React.createElement(ChatLauncherButton),
        React.createElement(ChatPanel),
        React.createElement(MobileChatDrawer)
      );

      const root = ReactDOM.createRoot(chatkitApp);
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
    // Add a slight delay to ensure Docusaurus is fully loaded
    setTimeout(initChatKit, 500);
  }
}