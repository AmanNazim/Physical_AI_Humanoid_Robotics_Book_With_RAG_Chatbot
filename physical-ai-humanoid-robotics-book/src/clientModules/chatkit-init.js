// Client-side initialization for ChatKit UI
export function loadChatKit() {
  // Load styles
  import('../../../rag_chatbot/chatkit/styles/variables.css');
  import('../../../rag_chatbot/chatkit/styles/theme.css');
  import('../../../rag_chatbot/chatkit/styles/breakpoints.css');
  import('../../../rag_chatbot/chatkit/styles/animations.css');

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
}