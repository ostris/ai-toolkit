import path from 'path';
export const TOOLKIT_ROOT = path.resolve('@', '..', '..');

const envPath = (name: string, fallback: string): string => {
  const value = process.env[name]?.trim();
  return value || fallback;
};

export const defaultTrainFolder = envPath('TRAINING_FOLDER', path.join(TOOLKIT_ROOT, 'output'));
export const defaultDatasetsFolder = envPath('DATASETS_FOLDER', path.join(TOOLKIT_ROOT, 'datasets'));
export const defaultDataRoot = envPath('DATA_ROOT', path.join(TOOLKIT_ROOT, 'data'));
export const defaultModelsFolder = envPath('MODELS_PATH', path.join(TOOLKIT_ROOT, 'models'));
