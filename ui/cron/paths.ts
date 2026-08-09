import path from 'path';
import prisma from './prisma';

export const TOOLKIT_ROOT = path.resolve('@', '..', '..');

const envPath = (name: string, fallback: string): string => {
  const value = process.env[name]?.trim();
  return value || fallback;
};

export const defaultTrainFolder = envPath('TRAINING_FOLDER', path.join(TOOLKIT_ROOT, 'output'));
export const defaultDatasetsFolder = envPath('DATASETS_FOLDER', path.join(TOOLKIT_ROOT, 'datasets'));
export const defaultDataRoot = envPath('DATA_ROOT', path.join(TOOLKIT_ROOT, 'data'));
export const defaultModelsFolder = envPath('MODELS_PATH', path.join(TOOLKIT_ROOT, 'models'));

// Forked file-server workers set AI_TOOLKIT_QUIET_PATHS so this line prints
// once per launched process group, not once per worker.
if (!process.env.AI_TOOLKIT_QUIET_PATHS) {
  console.log('TOOLKIT_ROOT:', TOOLKIT_ROOT);
}

export const getTrainingFolder = async () => {
  if (process.env.TRAINING_FOLDER?.trim()) {
    return process.env.TRAINING_FOLDER.trim();
  }

  const key = 'TRAINING_FOLDER';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let trainingRoot = defaultTrainFolder;
  if (row?.value && row.value !== '') {
    trainingRoot = row.value;
  }
  return trainingRoot as string;
};

export const getHFToken = async () => {
  if (process.env.HF_TOKEN?.trim()) {
    return process.env.HF_TOKEN.trim();
  }

  const key = 'HF_TOKEN';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let token = '';
  if (row?.value && row.value !== '') {
    token = row.value;
  }
  return token;
};

export const getModelsPath = async () => {
  if (process.env.MODELS_PATH?.trim()) {
    return process.env.MODELS_PATH.trim();
  }

  const key = 'MODELS_PATH';
  let row = await prisma.settings.findFirst({
    where: {
      key: key,
    },
  });
  let modelsPath = '';
  if (row?.value && row.value !== '' && row.value !== defaultModelsFolder) {
    modelsPath = row.value;
  }
  return modelsPath;
};

export const getDatabasePath = (): string => {
  const databaseUrl = process.env.DATABASE_URL?.trim();
  if (databaseUrl?.startsWith('file:/')) {
    return decodeURIComponent(databaseUrl.slice('file:'.length));
  }
  return path.join(TOOLKIT_ROOT, 'aitk_db.db');
};
