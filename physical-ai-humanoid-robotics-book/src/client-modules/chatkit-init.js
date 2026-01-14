// Client module to initialize ChatKit UI
// This file is loaded on the client-side after the page loads

if (typeof window !== 'undefined') {
  // Initialize ChatKit when the page is ready
  const initChatKit = async () => {
    console.log('Attempting to initialize ChatKit...');

    try {
      // Load ChatKit styles - correct relative paths from the book directory
      await Promise.all([
        import('@site/../rag_chatbot/chatkit/styles/variables.css'),
        import('@site/../rag_chatbot/chatkit/styles/theme.css'),
        import('@site/../rag_chatbot/chatkit/styles/breakpoints.css'),
        import('@site/../rag_chatbot/chatkit/styles/animations.css')
      ]);

      console.log('ChatKit styles loaded successfully');

      // Load React and ReactDOM directly since they may not be available globally in Docusaurus
      const [reactModule, reactDOMModule] = await Promise.all([
        import('react'),
        import('react-dom/client')
      ]);

      const React = reactModule.default || reactModule;
      const ReactDOM = reactDOMModule.default || reactDOMModule;

      console.log('React and ReactDOM loaded successfully');

      // Load ChatKit components
      console.log('Loading ChatKit components...');
      const chatkitModule = await import('@site/../rag_chatbot/chatkit');
      const { PortalManager, ChatLauncherButton, ChatPanel, MobileChatDrawer } = chatkitModule;

      console.log('ChatKit components loaded successfully', {
        hasPortalManager: !!PortalManager,
        hasChatLauncherButton: !!ChatLauncherButton,
        hasChatPanel: !!ChatPanel,
        hasMobileChatDrawer: !!MobileChatDrawer
      });

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

      console.log('ChatKit UI rendered successfully');
    } catch (error) {
      console.error('Error initializing ChatKit UI:', error);
    }
  };

  // Initialize when the page is ready
  if (document.readyState === 'loading') {
    console.log('DOM still loading, waiting for DOMContentLoaded');
    document.addEventListener('DOMContentLoaded', initChatKit);
  } else {
    console.log('DOM already loaded, initializing ChatKit after delay');
    // Add a slight delay to ensure Docusaurus is fully loaded
    setTimeout(initChatKit, 500);
  }
}