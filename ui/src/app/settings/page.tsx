'use client';

import { useEffect, useState } from 'react';
import useSettings from '@/hooks/useSettings';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { useLanguage } from '@/components/LanguageProvider';

export default function Settings() {
  const { settings, setSettings } = useSettings();
  const { locale, setLocale, t } = useLanguage();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [languages, setLanguages] = useState<{ locale: string; name: string }[]>([{ locale: 'en_US', name: 'English' }]);

  useEffect(() => {
    apiClient
      .get('/api/lang')
      .then(res => setLanguages(res.data))
      .catch(error => console.error('Error fetching languages:', error));
  }, []);

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
          <h1 className="text-base sm:text-lg">{t('settings.title')}</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <div className="space-y-4">
                <div>
                  <label htmlFor="language" className="block text-sm font-medium mb-2">
                    {t('common.language')}
                  </label>
                  <select
                    id="language"
                    value={locale}
                    onChange={e => setLocale(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                  >
                    {languages.map(language => (
                      <option key={language.locale} value={language.locale}>
                        {language.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">
                    {t('settings.huggingFaceToken')}
                    <div className="text-gray-500 text-sm ml-1">
                      {t('settings.huggingFaceTokenHelp')}{' '}
                      <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                        Huggingface
                      </a>
                    </div>
                  </label>
                  <input
                    type="password"
                    id="HF_TOKEN"
                    name="HF_TOKEN"
                    value={settings.HF_TOKEN}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder={t('settings.huggingFaceTokenPlaceholder')}
                  />
                </div>

                <div>
                  <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">
                    {t('settings.trainingFolderPath')}
                    <div className="text-gray-500 text-sm ml-1">
                      {t('settings.trainingFolderHelp')}
                    </div>
                  </label>
                  <input
                    type="text"
                    id="TRAINING_FOLDER"
                    name="TRAINING_FOLDER"
                    value={settings.TRAINING_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder={t('settings.trainingFolderPlaceholder')}
                  />
                </div>

                <div>
                  <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">
                    {t('settings.datasetFolderPath')}
                    <div className="text-gray-500 text-sm ml-1">
                      {t('settings.datasetFolderHelp')}{' '}
                      <span className="text-orange-800">{t('settings.datasetFolderWarning')}</span>
                    </div>
                  </label>
                  <input
                    type="text"
                    id="DATASETS_FOLDER"
                    name="DATASETS_FOLDER"
                    value={settings.DATASETS_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder={t('settings.datasetFolderPlaceholder')}
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
            {status === 'saving' ? t('common.saving') : t('common.saveSettings')}
          </button>

          {status === 'success' && <p className="text-green-500 text-center">{t('settings.saved')}</p>}
          {status === 'error' && <p className="text-red-500 text-center">{t('settings.saveError')}</p>}
        </form>
      </MainContent>
    </>
  );
}
