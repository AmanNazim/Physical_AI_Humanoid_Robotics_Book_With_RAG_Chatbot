import { useChatConversation } from '../contexts/ChatConversationContext';

export const useChatMessages = () => {
  const context = useChatConversation();
  if (!context) {
    throw new Error('useChatMessages must be used within a ChatConversationProvider');
  }
  return context;
};

export default useChatMessages;