// API Configuration for ChatKit
// This should be updated with your actual Hugging Face Space URL

// Default backend URL - replace with your actual Hugging Face Space URL
const DEFAULT_BACKEND_URL = 'https://aman778-rag-chatbot-backend.hf.space';

// Ensure HTTPS is used to avoid mixed content issues
const ensureHttps = (url) => {
  if (typeof url === 'string') {
    return url.replace(/^http:/, 'https:');
  }
  return url;
};

// Get backend URL from environment or use default
export const getBackendUrl = () => {
  if (typeof window !== 'undefined') {
    // In browser environment
    const envUrl = window.ENV?.REACT_APP_BACKEND_URL ||
      window.ENV?.BACKEND_URL ||
      (typeof process !== 'undefined' && process.env?.REACT_APP_BACKEND_URL) ||
      (typeof process !== 'undefined' && process.env?.BACKEND_URL) ||
      DEFAULT_BACKEND_URL;

    return ensureHttps(envUrl);
  }
  // For server-side rendering or other environments
  return ensureHttps(DEFAULT_BACKEND_URL);
};

export const BACKEND_URL = getBackendUrl();