'use client';

import { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { options, modelArchs, isVideoModelFromArch } from './options';
import { defaultJobConfig, defaultDatasetConfig } from './jobConfig';
import { JobConfig } from '@/types';
import { objectCopy } from '@/utils/basic';
import { useNestedState } from '@/utils/hooks';
import { TextInput, SelectInput, Checkbox, FormGroup, NumberInput } from '@/components/formInputs';
import Card from '@/components/Card';
import { X } from 'lucide-react';
import useSettings from '@/hooks/useSettings';
import useGPUInfo from '@/hooks/useGPUInfo';
import useDatasetList from '@/hooks/useDatasetList';
import path from 'path';
import { TopBar, MainContent } from '@/components/layout';
import { Button } from '@headlessui/react';
import { FaChevronLeft } from 'react-icons/fa';
import SimpleJob from './SimpleJob';
import AdvancedJob from './AdvancedJob';
import ErrorBoundary from '@/components/ErrorBoundary';
import { apiClient } from '@/utils/api';

const isDev = process.env.NODE_ENV === 'development';

export default function TrainingForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const runId = searchParams.get('id');
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const { settings, isSettingsLoaded } = useSettings();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const { datasets, status: datasetFetchStatus } = useDatasetList();
  const [datasetOptions, setDatasetOptions] = useState<{ value: string; label: string }[]>([]);
  const [showAdvancedView, setShowAdvancedView] = useState(false);

  const [jobConfig, setJobConfig] = useNestedState<JobConfig>(objectCopy(defaultJobConfig));
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  useEffect(() => {
    if (!isSettingsLoaded) return;
    if (datasetFetchStatus !== 'success') return;

    const datasetOptions = datasets.map(name => ({ value: path.join(settings.DATASETS_FOLDER, name), label: name }));
    setDatasetOptions(datasetOptions);
    const defaultDatasetPath = defaultDatasetConfig.folder_path;

    for (let i = 0; i < jobConfig.config.process[0].datasets.length; i++) {
      const dataset = jobConfig.config.process[0].datasets[i];
      if (dataset.folder_path === defaultDatasetPath) {
        if (datasetOptions.length > 0) {
          setJobConfig(datasetOptions[0].value, `config.process[0].datasets[${i}].folder_path`);
        }
      }
    }
  }, [datasets, settings, isSettingsLoaded, datasetFetchStatus]);

  useEffect(() => {
    if (runId) {
      apiClient
        .get(`/api/jobs?id=${runId}`)
        .then(res => res.data)
        .then(data => {
          console.log('Training:', data);
          setGpuIDs(data.gpu_ids);
          setJobConfig(JSON.parse(data.job_config));
        })
        .catch(error => console.error('Error fetching training:', error));
    }
  }, [runId]);

  useEffect(() => {
    if (isGPUInfoLoaded) {
      if (gpuIDs === null && gpuList.length > 0) {
        setGpuIDs(`${gpuList[0].index}`);
      }
    }
  }, [gpuList, isGPUInfoLoaded]);

  useEffect(() => {
    if (isSettingsLoaded) {
      setJobConfig(settings.TRAINING_FOLDER, 'config.process[0].training_folder');
    }
  }, [settings, isSettingsLoaded]);

  const saveJob = async () => {
    if (status === 'saving') return;
    setStatus('saving');

    apiClient
      .post('/api/jobs', {
        id: runId,
        name: jobConfig.config.name,
        gpu_ids: gpuIDs,
        job_config: jobConfig,
      })
      .then(res => {
        setStatus('success');
        if (runId) {
          router.push(`/jobs/${runId}`);
        } else {
          router.push(`/jobs/${res.data.id}`);
        }
      })
      .catch(error => {
        if (error.response?.status === 409) {
          alert('Training name already exists. Please choose a different name.');
        } else {
          alert('Failed to save job. Please try again.');
        }
        console.log('Error saving training:', error);
      })
      .finally(() =>
        setTimeout(() => {
          setStatus('idle');
        }, 2000),
      );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    saveJob();
  };

  return (
    <>
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">{runId ? 'Edit Training Job' : 'New Training Job'}</h1>
        </div>
        <div className="flex-1"></div>
        {showAdvancedView && (
          <>
            <div>
              <SelectInput
                value={`${gpuIDs}`}
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            </div>
            <div className="mx-4 bg-gray-200 dark:bg-gray-800 w-1 h-6"></div>
          </>
        )}

        <div className="pr-2">
          <Button
            className="text-gray-200 bg-gray-800 px-3 py-1 rounded-md"
            onClick={() => setShowAdvancedView(!showAdvancedView)}
          >
            {showAdvancedView ? 'Show Simple' : 'Show Advanced'}
          </Button>
        </div>
        <div>
          <Button
            className="text-gray-200 bg-green-800 px-3 py-1 rounded-md"
            onClick={() => saveJob()}
            disabled={status === 'saving'}
          >
            {status === 'saving' ? 'Saving...' : runId ? 'Update Job' : 'Create Job'}
          </Button>
        </div>
      </TopBar>

      {showAdvancedView ? (
        <div className="pt-[48px] absolute top-0 left-0 w-full h-full overflow-auto">
          <AdvancedJob
            jobConfig={jobConfig}
            setJobConfig={setJobConfig}
            status={status}
            handleSubmit={handleSubmit}
            runId={runId}
            gpuIDs={gpuIDs}
            setGpuIDs={setGpuIDs}
            gpuList={gpuList}
            datasetOptions={datasetOptions}
            settings={settings}
          />
        </div>
      ) : (
        <MainContent>
          <ErrorBoundary fallback={
            <div className="flex items-center justify-center h-64 text-lg text-red-600 font-medium bg-red-100 dark:bg-red-900/20 dark:text-red-400 border border-red-300 dark:border-red-700 rounded-lg">
              Advanced job detected. Please switch to advanced view to continue.
            </div>
          }>
            <SimpleJob
              jobConfig={jobConfig}
              setJobConfig={setJobConfig}
              status={status}
              handleSubmit={handleSubmit}
              runId={runId}
              gpuIDs={gpuIDs}
              setGpuIDs={setGpuIDs}
              gpuList={gpuList}
              datasetOptions={datasetOptions}
            />
          </ErrorBoundary>

          <div className="pt-20"></div>
        </MainContent>
      )}
    </>
  );
}