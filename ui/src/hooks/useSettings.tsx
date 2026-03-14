'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface Settings {
  HF_TOKEN: string;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
  COMFYUI_URL: string;
  COMFYUI_INPUT_DIR: string;
  COMFYUI_OUTPUT_DIR: string;
}

export default function useSettings() {
  const [settings, setSettings] = useState({
    HF_TOKEN: '',
    TRAINING_FOLDER: '',
    DATASETS_FOLDER: '',
    COMFYUI_URL: '',
    COMFYUI_INPUT_DIR: '',
    COMFYUI_OUTPUT_DIR: '',
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    apiClient
      .get('/api/settings')
      .then(res => res.data)
      .then(data => {
        console.log('Settings:', data);
        setSettings({
          HF_TOKEN: data.HF_TOKEN || '',
          TRAINING_FOLDER: data.TRAINING_FOLDER || '',
          DATASETS_FOLDER: data.DATASETS_FOLDER || '',
          COMFYUI_URL: data.COMFYUI_URL || '',
          COMFYUI_INPUT_DIR: data.COMFYUI_INPUT_DIR || '',
          COMFYUI_OUTPUT_DIR: data.COMFYUI_OUTPUT_DIR || '',
        });
        setIsLoaded(true);
      })
      .catch(error => console.error('Error fetching settings:', error));
  }, []);

  return { settings, setSettings, isSettingsLoaded };
}
