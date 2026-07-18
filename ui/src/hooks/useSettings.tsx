'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface Settings {
  HF_TOKEN: string;
  OFFLINE_MODE: boolean;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
}

export default function useSettings() {
  const [settings, setSettings] = useState<Settings>({
    HF_TOKEN: '',
    OFFLINE_MODE: false,
    TRAINING_FOLDER: '',
    DATASETS_FOLDER: '',
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    apiClient
      .get<Settings>('/api/settings')
      .then(res => res.data)
      .then(data => {
        console.log('Settings:', data);
        setSettings({
          HF_TOKEN: data.HF_TOKEN || '',
          OFFLINE_MODE: data.OFFLINE_MODE,
          TRAINING_FOLDER: data.TRAINING_FOLDER || '',
          DATASETS_FOLDER: data.DATASETS_FOLDER || '',
        });
        setIsLoaded(true);
      })
      .catch(error => console.error('Error fetching settings:', error));
  }, []);

  return { settings, setSettings, isSettingsLoaded };
}
