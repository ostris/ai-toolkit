'use client';

import { useEffect, useState } from 'react';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';

interface SettingField {
  key: string;
  label: string;
  input_type: string;
  description: string;
  placeholder: string;
}

interface Plugin {
  id: string;
  display_name: string;
  settings_schema: SettingField[];
}

export default function Settings() {
  const [values, setValues] = useState<Record<string, string>>({
    HF_TOKEN: '',
    TRAINING_FOLDER: '',
    DATASETS_FOLDER: '',
  });
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  useEffect(() => {
    apiClient
      .get('/api/settings')
      .then(res => setValues(res.data as Record<string, string>))
      .catch(err => console.error('Error fetching settings:', err));

    apiClient
      .get('/api/datasets/remote/plugins')
      .then(res => setPlugins(res.data as Plugin[]))
      .catch(err => console.error('Error fetching plugins:', err));
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, type, value, checked } = e.target;
    setValues(prev => ({ ...prev, [name]: type === 'checkbox' ? (checked ? 'true' : 'false') : value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('saving');

    apiClient
      .post('/api/settings', values)
      .then(() => setStatus('success'))
      .catch(() => setStatus('error'))
      .finally(() => setTimeout(() => setStatus('idle'), 2000));
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
                    value={values['HF_TOKEN'] ?? ''}
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
                    value={values['TRAINING_FOLDER'] ?? ''}
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
                    value={values['DATASETS_FOLDER'] ?? ''}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter datasets folder path"
                  />
                </div>

                {plugins.map(plugin =>
                  plugin.settings_schema.length === 0 ? null : (
                    <div key={plugin.id}>
                      <hr className="border-gray-700 my-2" />
                      <h2 className="text-sm font-semibold mb-3">{plugin.display_name}</h2>
                      <div className="space-y-4">
                        {plugin.settings_schema.map(field => (
                          <div key={field.key}>
                            {field.input_type === 'checkbox' ? (
                              <label className="flex items-center gap-3 text-sm font-medium cursor-pointer">
                                <input
                                  type="checkbox"
                                  id={field.key}
                                  name={field.key}
                                  checked={(values[field.key] ?? 'true') !== 'false'}
                                  onChange={handleChange}
                                  className="w-4 h-4 rounded border-gray-700 bg-gray-800 accent-blue-500"
                                />
                                <span>
                                  {field.label}
                                  {field.description && (
                                    <div className="text-gray-500 text-sm">{field.description}</div>
                                  )}
                                </span>
                              </label>
                            ) : (
                              <>
                                <label htmlFor={field.key} className="block text-sm font-medium mb-2">
                                  {field.label}
                                  {field.description && (
                                    <div className="text-gray-500 text-sm ml-1">{field.description}</div>
                                  )}
                                </label>
                                <input
                                  type={field.input_type}
                                  id={field.key}
                                  name={field.key}
                                  value={values[field.key] ?? ''}
                                  onChange={handleChange}
                                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                                  placeholder={field.placeholder}
                                />
                              </>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ),
                )}
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
      </MainContent>
    </>
  );
}

