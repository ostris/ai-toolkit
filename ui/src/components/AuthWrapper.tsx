'use client';

import { useState, useEffect, useRef } from 'react';
import { apiClient, isAuthorizedState } from '@/utils/api';
import { createGlobalState } from 'react-global-hooks';

interface AuthWrapperProps {
  authRequired: boolean;
  children: React.ReactNode | React.ReactNode[];
}

export default function AuthWrapper({ authRequired, children }: AuthWrapperProps) {
  const [token, setToken] = useState('');
  // start with true, and deauth if needed
  const [isAuthorizedGlobal, setIsAuthorized] = isAuthorizedState.use();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [isBrowser, setIsBrowser] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const isAuthorized = authRequired ? isAuthorizedGlobal : true;

  // Set isBrowser to true when component mounts
  useEffect(() => {
    setIsBrowser(true);
    // Get token from localStorage only after component has mounted
    const storedToken = localStorage.getItem('AI_TOOLKIT_AUTH') || '';
    setToken(storedToken);
    checkAuth();
  }, []);

  // auto focus on input when not authorized
  useEffect(() => {
    if (isAuthorized) {
      return;
    }
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus();
      }
    }, 100);
  }, [isAuthorized]);

  const checkAuth = async () => {
    // always get current stored token here to avoid state race conditions
    const currentToken = localStorage.getItem('AI_TOOLKIT_AUTH') || '';
    if (!authRequired || isLoading || currentToken === '') {
      return;
    }
    setIsLoading(true);
    setError('');
    try {
      const response = await apiClient.get('/api/auth');
      if (response.data.isAuthenticated) {
        setIsAuthorized(true);
      } else {
        setIsAuthorized(false);
        setError('Invalid token. Please try again.');
      }
    } catch (err) {
      setIsAuthorized(false);
      console.log(err);
      setError('Invalid token. Please try again.');
    }
    setIsLoading(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!token.trim()) {
      setError('Please enter your token');
      return;
    }

    if (isBrowser) {
      localStorage.setItem('AI_TOOLKIT_AUTH', token);
      checkAuth();
    }
  };

  if (isAuthorized) {
    return <>{children}</>;
  }

  return (
    <div className="flex min-h-screen bg-gray-900 text-gray-100 absolute top-0 left-0 right-0 bottom-0 scroll-auto">
      {/* Left side - decorative or brand area */}
      <div className="hidden lg:flex lg:w-1/2 bg-gray-800 flex-col justify-center items-center p-12">
        <div className="mb-4">
          {/* Replace with your own logo */}
          <div className="flex items-center justify-center">
            <img src="/ostris_logo.png" alt="Ostris AI Toolkit" className="w-auto h-24 inline" />
          </div>
        </div>
        <h1 className="text-4xl mb-6">AI Toolkit</h1>
      </div>

      {/* Right side - login form */}
      <div className="w-full lg:w-1/2 flex flex-col justify-center items-center p-8 sm:p-12">
        <div className="w-full max-w-md">
          <div className="lg:hidden flex justify-center mb-4">
            {/* Mobile logo */}
            <div className="flex items-center justify-center">
              <img src="/ostris_logo.png" alt="Ostris AI Toolkit" className="w-auto h-24 inline" />
            </div>
          </div>

          <h2 className="text-3xl text-center mb-2 lg:hidden">AI Toolkit</h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="token" className="block text-sm font-medium text-gray-400 mb-2">
                Access Token
              </label>
              <input
                id="token"
                name="token"
                type="password"
                autoComplete="off"
                required
                value={token}
                ref={inputRef}
                onChange={e => setToken(e.target.value)}
                className="w-full px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 text-gray-100 transition duration-200"
                placeholder="Enter your token"
              />
            </div>

            {error && (
              <div className="p-3 bg-red-900/50 border border-red-800 rounded-lg text-red-200 text-sm">{error}</div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition duration-200 flex items-center justify-center"
            >
              {isLoading ? (
                <svg
                  className="animate-spin h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
              ) : (
                'Check Token'
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
