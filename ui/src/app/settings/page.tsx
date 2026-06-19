'use client';

import { useEffect, useState } from 'react';
import useSettings from '@/hooks/useSettings';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';

export default function Settings() {
  const { settings, setSettings } = useSettings();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [restartState, setRestartState] = useState<'idle' | 'confirming' | 'restarting' | 'waiting' | 'back'>('idle');

  const handleRestart = async () => {
    if (restartState === 'idle') {
      setRestartState('confirming');
      return;
    }
    if (restartState !== 'confirming') return;
    setRestartState('restarting');
    try {
      await apiClient.post('/api/restart', {});
    } catch (err) {
      // Network error is expected — the server dies while sending the response.
    }
    setRestartState('waiting');

    // Poll for the server to come back. Look for a small, cheap endpoint.
    const start = Date.now();
    const maxWaitMs = 45_000;
    const probe = async (): Promise<boolean> => {
      try {
        const res = await fetch('/api/settings', { cache: 'no-store' });
        return res.ok;
      } catch {
        return false;
      }
    };
    // Give the supervisor (`concurrently`) a moment to relaunch.
    await new Promise(r => setTimeout(r, 1500));
    while (Date.now() - start < maxWaitMs) {
      if (await probe()) {
        setRestartState('back');
        setTimeout(() => window.location.reload(), 800);
        return;
      }
      await new Promise(r => setTimeout(r, 1000));
    }
    setRestartState('idle');
    alert(
      'The server did not come back online within 45 seconds.\n\n' +
        'If you launched via Start-AI-Toolkit.bat the supervisor auto-restarts the server, ' +
        'but if you launched it manually with `npm run dev` you will need to restart it yourself.',
    );
  };

  const restartLabel = (() => {
    switch (restartState) {
      case 'idle':
        return 'Restart Server';
      case 'confirming':
        return 'Click again to confirm restart';
      case 'restarting':
        return 'Restarting…';
      case 'waiting':
        return 'Waiting for server to come back…';
      case 'back':
        return 'Server is back — reloading…';
    }
  })();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('saving');

    apiClient
      .post('/api/settings', settings)
      .then(() => {
        setStatus('success');
      })
      .catch(error => {
        console.error('Error saving settings:', error);
        setStatus('error');
      })
      .finally(() => {
        setTimeout(() => setStatus('idle'), 2000);
      });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSettings(prev => ({ ...prev, [name]: value }));
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Settings</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <div className="space-y-4">
                <div>
                  <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">
                    Hugging Face Token
                    <div className="text-gray-500 text-sm ml-1">
                      Create a Read token on{' '}
                      <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                        {' '}
                        Huggingface
                      </a>{' '}
                      if you need to access gated/private models.
                    </div>
                  </label>
                  <input
                    type="password"
                    id="HF_TOKEN"
                    name="HF_TOKEN"
                    value={settings.HF_TOKEN}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter your Hugging Face token"
                  />
                </div>

                <div>
                  <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">
                    Training Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      We will store your training information here. Must be an absolute path. If blank, it will default
                      to the output folder in the project root.
                    </div>
                  </label>
                  <input
                    type="text"
                    id="TRAINING_FOLDER"
                    name="TRAINING_FOLDER"
                    value={settings.TRAINING_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter training folder path"
                  />
                </div>

                <div>
                  <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">
                    Dataset Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      Where we store and find your datasets.{' '}
                      <span className="text-orange-800">
                        Warning: This software may modify datasets so it is recommended you keep a backup somewhere else
                        or have a dedicated folder for this software.
                      </span>
                    </div>
                  </label>
                  <input
                    type="text"
                    id="DATASETS_FOLDER"
                    name="DATASETS_FOLDER"
                    value={settings.DATASETS_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter datasets folder path"
                  />
                </div>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={status === 'saving'}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status === 'saving' ? 'Saving...' : 'Save Settings'}
          </button>

          {status === 'success' && <p className="text-green-500 text-center">Settings saved successfully!</p>}
          {status === 'error' && <p className="text-red-500 text-center">Error saving settings. Please try again.</p>}
        </form>

        <div className="mt-10 pt-6 border-t border-gray-800">
          <h2 className="text-lg font-medium mb-1">Server</h2>
          <p className="text-sm text-gray-400 mb-3">
            Gracefully exit the Next.js process. The launcher's supervisor (
            <code className="bg-gray-800 px-1 rounded">concurrently</code>) auto-restarts it within ~1 second.
            This page will reload automatically once the server is back.
          </p>
          <button
            type="button"
            onClick={handleRestart}
            disabled={restartState !== 'idle' && restartState !== 'confirming'}
            className={`px-4 py-2 rounded-lg transition-colors disabled:cursor-wait ${
              restartState === 'confirming'
                ? 'bg-amber-600 hover:bg-amber-500 text-white'
                : restartState === 'idle'
                  ? 'bg-red-700 hover:bg-red-600 text-white'
                  : 'bg-gray-700 text-gray-200'
            }`}
          >
            {restartLabel}
          </button>
          {restartState === 'confirming' && (
            <button
              type="button"
              onClick={() => setRestartState('idle')}
              className="ml-2 px-3 py-2 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded-lg"
            >
              Cancel
            </button>
          )}
        </div>
      </MainContent>
    </>
  );
}
