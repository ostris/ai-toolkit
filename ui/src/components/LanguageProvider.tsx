'use client';

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { apiClient } from '@/utils/api';

type Messages = Record<string, string>;

interface LanguageContextValue {
  locale: string;
  setLocale: (locale: string) => void;
  t: (key: string, fallback?: string) => string;
}

const DEFAULT_LOCALE = 'en_US';
const STORAGE_KEY = 'AI_TOOLKIT_LANG';

const defaultMessages: Messages = {
  'common.language': 'Language',
  'common.languageName': 'English',
  'common.english': 'English',
  'common.chinese': 'Chinese',
  'common.french': 'French',
  'common.esperanto': 'Esperanto',
  'common.saveSettings': 'Save Settings',
  'common.saving': 'Saving...',
  'common.viewAll': 'View All',
  'common.loading': 'Loading...',
  'common.error': 'Error',
  'common.empty': 'Empty',
  'common.refresh': 'Refresh',
  'common.cancel': 'Cancel',
  'common.confirm': 'Confirm',
  'common.create': 'Create',
  'common.delete': 'Delete',
  'common.clear': 'Clear',
  'common.close': 'Close',
  'common.actions': 'Actions',
  'common.name': 'Name',
  'common.status': 'Status',
  'common.info': 'Info',
  'common.simple': 'Simple',
  'common.advanced': 'Advanced',
  'nav.dashboard': 'Dashboard',
  'nav.newJob': 'New Job',
  'nav.queue': 'Queue',
  'nav.datasets': 'Datasets',
  'nav.settings': 'Settings',
  'nav.support': 'Support AI-Toolkit',
  'settings.title': 'Settings',
  'settings.huggingFaceToken': 'Hugging Face Token',
  'settings.huggingFaceTokenHelp':
    'Create a Read token on Huggingface if you need to access gated/private models.',
  'settings.huggingFaceTokenPlaceholder': 'Enter your Hugging Face token',
  'settings.trainingFolderPath': 'Training Folder Path',
  'settings.trainingFolderHelp':
    'We will store your training information here. Must be an absolute path. If blank, it will default to the output folder in the project root.',
  'settings.trainingFolderPlaceholder': 'Enter training folder path',
  'settings.datasetFolderPath': 'Dataset Folder Path',
  'settings.datasetFolderHelp': 'Where we store and find your datasets.',
  'settings.datasetFolderWarning':
    'Warning: This software may modify datasets so it is recommended you keep a backup somewhere else or have a dedicated folder for this software.',
  'settings.datasetFolderPlaceholder': 'Enter datasets folder path',
  'settings.saved': 'Settings saved successfully!',
  'settings.saveError': 'Error saving settings. Please try again.',
  'dashboard.title': 'Dashboard',
  'dashboard.queues': 'Queues',
  'jobs.queueTitle': 'Queue',
  'jobs.newJobShort': '+ New Job',
  'jobs.newTrainingJob': 'New Training Job',
  'jobs.editTrainingJob': 'Edit Training Job',
  'jobs.importConfig': 'Import Config',
  'jobs.showSimple': 'Show Simple',
  'jobs.showAdvanced': 'Show Advanced',
  'jobs.update': 'Update',
  'jobs.create': 'Create',
  'jobs.updateJob': 'Update Job',
  'jobs.createJob': 'Create Job',
  'jobs.nameExists': 'Training name already exists. Please choose a different name.',
  'jobs.saveFailed': 'Failed to save job. Please try again.',
  'jobs.parseFailed': 'Failed to parse config file. Please check the file format.',
  'jobs.advancedDetected': 'Advanced job detected. Please switch to advanced view to continue.',
  'jobs.tableName': 'Name',
  'jobs.steps': 'Steps',
  'jobs.gpu': 'GPU',
  'jobs.deleteJobs': 'Delete Jobs',
  'jobs.deleteJobsMessage': 'Are you sure you want to delete {count} job{plural}? This will also permanently remove them from your disk.',
  'jobs.deleteJobsRunningWarning': ' WARNING: {count} of them {verb} currently running and will be stopped first.',
  'jobs.deletingProgress': 'Deleting {done} / {total}...',
  'jobs.selectedCount': '{count} job{plural} selected',
  'jobs.deleteSelected': 'Delete Selected',
  'jobs.queueRunning': 'Queue Running',
  'jobs.queueStopped': 'Queue Stopped',
  'jobs.stop': 'STOP',
  'jobs.start': 'START',
  'jobs.idle': 'Idle',
  'jobDetail.overview': 'Overview',
  'jobDetail.samples': 'Samples',
  'jobDetail.lossGraph': 'Loss Graph',
  'jobDetail.configFile': 'Config File',
  'jobDetail.plugin': 'Plugin',
  'jobDetail.jobTitle': 'Job: {name}',
  'jobDetail.captioningTitle': 'Captioning: {name}',
  'jobDetail.loadingJob': 'Loading...',
  'jobDetail.fetchError': 'Error fetching job',
  'datasets.title': 'Datasets',
  'datasets.datasetName': 'Dataset Name',
  'datasets.newShort': '+ New',
  'datasets.newDataset': 'New Dataset',
  'datasets.deleteDataset': 'Delete Dataset',
  'datasets.deleteDatasetMessage': 'Are you sure you want to delete the dataset "{name}"? This action cannot be undone.',
  'datasets.newDatasetMessage': 'Enter the name of the new dataset:',
  'datasets.nameRequired': 'Dataset name is required.',
  'datasets.createFolderHelp': 'This will create a new folder with the name below in your dataset folder.',
  'datasetDetail.title': 'Dataset: {name}',
  'datasetDetail.captionExt': 'Caption ext',
  'datasetDetail.addShort': '+ Add',
  'datasetDetail.addImages': 'Add Images',
  'datasetDetail.loadingImages': 'Loading Images',
  'datasetDetail.loadingImagesHelp': 'Please wait while we fetch your dataset images...',
  'datasetDetail.errorLoadingImages': 'Error Loading Images',
  'datasetDetail.errorLoadingImagesHelp': 'There was a problem fetching the images. Please try refreshing the page.',
  'datasetDetail.noImagesFound': 'No Images Found',
  'datasetDetail.noImagesFoundHelp': 'This dataset is empty. Click "Add Images" to get started.',
  'monitor.gpuMonitor': 'GPU Monitor',
  'monitor.lastUpdated': 'Last updated',
  'monitor.errorBang': 'Error!',
  'monitor.noGpuData': 'No GPU data available.',
  'monitor.noNvidiaGpu': 'No NVIDIA GPUs detected!',
  'monitor.noNvidiaSmi': 'nvidia-smi is not available on this system.',
  'monitor.noGpuFound': 'No GPUs found, but nvidia-smi is available.',
  'monitor.temperature': 'Temperature',
  'monitor.fanSpeed': 'Fan Speed',
  'monitor.gpuLoad': 'GPU Load',
  'monitor.memory': 'Memory',
  'monitor.clockSpeed': 'Clock Speed',
  'monitor.powerDraw': 'Power Draw',
  'monitor.cpuInfo': 'CPU Info',
  'monitor.noCpuData': 'No CPU data available',
  'monitor.cpuLoad': 'CPU Load',
  'dataset.autoCaptioning': 'Auto Captioning...',
  'dataset.captioning': 'Captioning',
  'dataset.autoCaption': 'Auto Caption',
  'dataset.caption': 'Caption',
  'dataset.captionDataset': 'Caption Dataset',
  'dataset.datasetPathMissing': 'Dataset path is missing. Please try again.',
  'dataset.captionJobExists': 'A caption job for this dataset already exists. Please check your jobs list.',
  'dataset.addToQueue': 'Add to Queue',
  'dataset.addImagesTo': 'Add Images to: {name}',
  'dataset.dropFiles': 'Drag & drop files here or click to select',
  'dataset.supportedFiles': 'Images, videos, .txt or .json supported',
  'dataset.dropMoreFiles': 'Drop more files to add to queue',
  'dataset.uploading': 'Uploading...',
  'dataset.uploadFailed': 'Upload failed',
  'dataset.filesFailed': '{count} files failed',
  'dataset.cancelUpload': 'Cancel Upload',
  'dataset.failed': 'Failed',
  'dataset.queued': 'Queued',
  'dataset.deleteImage': 'Delete Image',
  'dataset.deleteVideo': 'Delete Video',
  'dataset.deleteImageMessage': 'Are you sure you want to delete this image? This action cannot be undone.',
  'dataset.deleteVideoMessage': 'Are you sure you want to delete this video? This action cannot be undone.',
  'dataset.loadingCaption': 'Loading caption...',
  'dataset.hideBoundingBoxes': 'Hide bounding boxes',
  'dataset.showEditBoundingBoxes': 'Show & edit bounding boxes',
  'dataset.download': 'Download',
  'dataset.file': 'File',
  'dataset.templates': 'Templates',
  'dataset.addCaption': 'Add a caption...',
  'jobActions.stopJob': 'Stop Job',
  'jobActions.stopJobMessage': 'Are you sure you want to stop the job "{name}"? You CAN resume later.',
  'jobActions.stop': 'Stop',
  'jobActions.deleteJob': 'Delete Job',
  'jobActions.deleteJobMessage': 'Are you sure you want to delete the job "{name}"? This will also permanently remove it from your disk.',
  'jobActions.deleteRunningWarning': 'WARNING: The job is currently running. You should stop it first if you can.',
  'jobActions.cloneJob': 'Clone Job',
  'jobActions.saveNextStep': 'Save Next Step',
  'jobActions.markStopped': 'Mark Job as Stopped',
  'jobActions.markStoppedMessage':
    "Are you sure you want to mark this job as stopped? This will set the job status to 'stopped' if the status is hung. Only do this if you are 100% sure the job is stopped. This will NOT stop the job.",
  'jobOverview.progress': 'Progress',
  'jobOverview.stepOf': 'Step {step} of {total}',
  'jobOverview.jobName': 'Job Name',
  'jobOverview.assignedGpus': 'Assigned GPUs',
  'jobOverview.gpus': 'GPUs',
  'jobOverview.speed': 'Speed',
  'jobOverview.loadingLog': 'Loading log...',
  'jobOverview.errorLoadingLog': 'Error loading log',
  'jobInfo.Training completed': 'Training completed',
  'jobInfo.Training started': 'Training started',
  'jobInfo.Training stopped': 'Training stopped',
  'jobInfo.Training failed': 'Training failed',
  'status.running': 'running',
  'status.stopping': 'stopping',
  'status.stopped': 'stopped',
  'status.completed': 'completed',
  'status.failed': 'failed',
  'status.queued': 'queued',
  'status.error': 'error',
};

const LanguageContext = createContext<LanguageContextValue>({
  locale: DEFAULT_LOCALE,
  setLocale: () => {},
  t: (key, fallback) => defaultMessages[key] ?? fallback ?? key,
});

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState(DEFAULT_LOCALE);
  const [messages, setMessages] = useState<Messages>(defaultMessages);

  useEffect(() => {
    const storedLocale = localStorage.getItem(STORAGE_KEY);
    if (storedLocale) {
      setLocaleState(storedLocale);
    }
  }, []);

  useEffect(() => {
    document.documentElement.lang = locale;
    localStorage.setItem(STORAGE_KEY, locale);

    if (locale === DEFAULT_LOCALE) {
      setMessages(defaultMessages);
      return;
    }

    apiClient
      .get(`/api/lang/${encodeURIComponent(locale)}`)
      .then(res => {
        setMessages({ ...defaultMessages, ...res.data });
      })
      .catch(error => {
        console.error(`Error loading language ${locale}:`, error);
        setMessages(defaultMessages);
      });
  }, [locale]);

  const setLocale = useCallback((nextLocale: string) => {
    setLocaleState(nextLocale);
  }, []);

  const value = useMemo<LanguageContextValue>(
    () => ({
      locale,
      setLocale,
      t: (key: string, fallbackText?: string) => {
        const fallback = defaultMessages[key] ?? fallbackText ?? key;
        const translated = messages[key] ?? fallback;

        return translated;
      },
    }),
    [locale, messages, setLocale],
  );

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  return useContext(LanguageContext);
}

export function formatMessage(message: string, values: Record<string, string | number>) {
  return message.replace(/\{(\w+)\}/g, (_match, key) => `${values[key] ?? ''}`);
}
