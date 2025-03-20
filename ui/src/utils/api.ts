import axios from 'axios';
import { createGlobalState } from 'react-global-hooks';

export const isAuthorizedState = createGlobalState(false);

export const apiClient = axios.create();

// Add a request interceptor to add token from localStorage
apiClient.interceptors.request.use(config => {
  const token = localStorage.getItem('AI_TOOLKIT_AUTH');
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  return config;
});

// Add a response interceptor to handle 401 errors
apiClient.interceptors.response.use(
  response => response, // Return successful responses as-is
  error => {
    // Check if the error is a 401 Unauthorized
    if (error.response && error.response.status === 401) {
      // Clear the auth token from localStorage
      localStorage.removeItem('AI_TOOLKIT_AUTH');
      isAuthorizedState.set(false);
    }

    // Reject the promise with the error so calling code can still catch it
    return Promise.reject(error);
  },
);
