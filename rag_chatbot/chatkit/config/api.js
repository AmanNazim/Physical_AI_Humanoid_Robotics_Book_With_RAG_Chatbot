// API Configuration for ChatKit
// This should be updated with your actual Hugging Face Space URL

// Default backend URL - replace with your actual Hugging Face Space URL
const DEFAULT_BACKEND_URL = 'https://aman778-rag-chatbot-backend.hf.space';

// Get backend URL from environment or use default
export const getBackendUrl = () => {
  if (typeof window !== 'undefined') {
    // In browser environment
    return (
      window.ENV?.REACT_APP_BACKEND_URL ||
      window.ENV?.BACKEND_URL ||
      (typeof process !== 'undefined' && process.env?.REACT_APP_BACKEND_URL) ||
      (typeof process !== 'undefined' && process.env?.BACKEND_URL) ||
      DEFAULT_BACKEND_URL
    );
  }
  // For server-side rendering or other environments
  return DEFAULT_BACKEND_URL;
};

export const BACKEND_URL = getBackendUrl();