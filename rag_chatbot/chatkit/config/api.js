// API Configuration for ChatKit
// This should be updated with your actual Hugging Face Space URL

// Default backend URL - replace with your actual Hugging Face Space URL
const DEFAULT_BACKEND_URL = 'https://aman778-rag-chatbot-backend.hf.space';

// Ensure HTTPS is used to avoid mixed content issues
const ensureHttps = (url) => {
  if (typeof url === 'string') {
    // Handle various URL formats
    if (url.startsWith('http://')) {
      return url.replace('http://', 'https://');
    } else if (url.startsWith('//')) {
      // Handle protocol-relative URLs
      return 'https:' + url;
    } else if (!url.startsWith('https://') && url.includes('://')) {
      // If it has another protocol scheme, replace the scheme
      return url.replace(/^[^:]+:\/\//, 'https://');
    }
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

    console.log('DEBUG: Raw envUrl:', envUrl);
    const result = ensureHttps(envUrl);
    console.log('DEBUG: Final URL after ensureHttps:', result);
    return result;
  }
  // For server-side rendering or other environments
  const result = ensureHttps(DEFAULT_BACKEND_URL);
  console.log('DEBUG: Server-side URL after ensureHttps:', result);
  return result;
};

// Export a getter function instead of a constant to ensure it's evaluated at runtime
export const getBackendURL = () => {
  return getBackendUrl();
};