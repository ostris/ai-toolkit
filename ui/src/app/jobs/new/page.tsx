'use client';

import { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { defaultJobConfig, defaultDatasetConfig, migrateJobConfig } from './jobConfig';
import { jobTypeOptions } from './options';
import { JobConfig } from '@/types';
import { objectCopy } from '@/utils/basic';
import { useNestedState } from '@/utils/hooks';
import { SelectInput } from '@/components/formInputs';
import useSettings from '@/hooks/useSettings';
import useGPUInfo from '@/hooks/useGPUInfo';
import useDatasetList from '@/hooks/useDatasetList';
import path from 'path';
import { TopBar, MainContent } from '@/components/layout';
import { Button } from '@headlessui/react';
import { FaChevronLeft, FaSave, FaSlidersH, FaList, FaLayerGroup } from 'react-icons/fa';
import SimpleJob from './SimpleJob';
import AdvancedJob from './AdvancedJob';
import ErrorBoundary from '@/components/ErrorBoundary';
import { apiClient } from '@/utils/api';

const isDev = process.env.NODE_ENV === 'development';

export default function TrainingForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const runId = searchParams.get('id');
  const cloneId = searchParams.get('cloneId');
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

  // clone existing job
  useEffect(() => {
    if (cloneId) {
      apiClient
        .get(`/api/jobs?id=${cloneId}`)
        .then(res => res.data)
        .then(data => {
          console.log('Clone Training:', data);
          setGpuIDs(data.gpu_ids);
          const newJobConfig = migrateJobConfig(JSON.parse(data.job_config));
          newJobConfig.config.name = `${newJobConfig.config.name}_copy`;
          setJobConfig(newJobConfig);
        })
        .catch(error => console.error('Error fetching training:', error));
    }
  }, [cloneId]);

  useEffect(() => {
    if (runId) {
      apiClient
        .get(`/api/jobs?id=${runId}`)
        .then(res => res.data)
        .then(data => {
          console.log('Training:', data);
          setGpuIDs(data.gpu_ids);
          setJobConfig(migrateJobConfig(JSON.parse(data.job_config)));
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
        <div className="hidden md:flex items-center h-full">
          <Button className="text-gray-500 dark:text-gray-300 px-2 flex items-center h-full" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div className="md:block hidden">
          <h1 className="text-lg whitespace-nowrap">{runId ? 'Edit Training Job' : 'New Training Job'}</h1>
        </div>
        <div className="md:hidden flex items-center flex-1 justify-center">
          <h1 className="text-lg whitespace-nowrap">{runId ? 'Edit Job' : 'New Job'}</h1>
        </div>
        <div className="md:flex-1"></div>
        {showAdvancedView && (
          <>
            <div className="hidden md:block">
              <SelectInput
                value={`${gpuIDs}`}
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            </div>
            <div className="hidden md:block mx-4 bg-gray-200 dark:bg-gray-800 w-1 h-6"></div>
          </>
        )}
        {!showAdvancedView && (
          <>
            <div className="hidden md:block">
              <SelectInput
                value={`${jobConfig?.config.process[0].type}`}
                onChange={value => {
                  // undo current job type changes
                  const currentOption = jobTypeOptions.find(
                    option => option.value === jobConfig?.config.process[0].type,
                  );
                  if (currentOption && currentOption.onDeactivate) {
                    setJobConfig(currentOption.onDeactivate(objectCopy(jobConfig)));
                  }
                  const option = jobTypeOptions.find(option => option.value === value);
                  if (option) {
                    if (option.onActivate) {
                      setJobConfig(option.onActivate(objectCopy(jobConfig)));
                    }
                    jobTypeOptions.forEach(opt => {
                      if (opt.value !== option.value && opt.onDeactivate) {
                        setJobConfig(opt.onDeactivate(objectCopy(jobConfig)));
                      }
                    });
                  }
                  setJobConfig(value, 'config.process[0].type');
                }}
                options={jobTypeOptions}
              />
            </div>
            {/* Mobile Icon-only dropdown trigger */}
            <div className="md:hidden relative group mr-2">
              <div
                className="text-gray-200 px-3 py-1 bg-gray-800 rounded-md cursor-pointer h-[30px] flex items-center justify-center"
                onClick={e => {
                  // Toggle a manual dropdown if we can't style the select properly
                  const dropdown = document.getElementById('mobile-job-type-dropdown');
                  const overlay = document.getElementById('mobile-dropdown-overlay');
                  if (dropdown && overlay) {
                    dropdown.classList.toggle('hidden');
                    overlay.classList.toggle('hidden');
                  }
                }}
              >
                <FaLayerGroup className="text-lg" />
              </div>
              {/* Custom Mobile Dropdown */}
              <div
                id="mobile-job-type-dropdown"
                className="hidden absolute right-0 top-full mt-2 w-auto min-w-[140px] bg-gray-800 border border-gray-700 rounded-md shadow-lg z-50"
              >
                {jobTypeOptions.map(opt => (
                  <div
                    key={opt.value}
                    className="px-4 py-2 text-sm text-gray-200 hover:bg-gray-700 cursor-pointer border-b border-gray-700 last:border-0 whitespace-nowrap"
                    onClick={() => {
                      const value = opt.value;
                      // undo current job type changes
                      const currentOption = jobTypeOptions.find(
                        option => option.value === jobConfig?.config.process[0].type,
                      );
                      if (currentOption && currentOption.onDeactivate) {
                        setJobConfig(currentOption.onDeactivate(objectCopy(jobConfig)));
                      }
                      const option = jobTypeOptions.find(option => option.value === value);
                      if (option) {
                        if (option.onActivate) {
                          setJobConfig(option.onActivate(objectCopy(jobConfig)));
                        }
                        jobTypeOptions.forEach(opt => {
                          if (opt.value !== option.value && opt.onDeactivate) {
                            setJobConfig(opt.onDeactivate(objectCopy(jobConfig)));
                          }
                        });
                      }
                      setJobConfig(value, 'config.process[0].type');
                      document.getElementById('mobile-job-type-dropdown')?.classList.add('hidden');
                      document.getElementById('mobile-dropdown-overlay')?.classList.add('hidden');
                    }}
                  >
                    {opt.label}
                  </div>
                ))}
              </div>
              {/* Overlay to close dropdown when clicking outside */}
              <div
                className="hidden fixed inset-0 z-40"
                id="mobile-dropdown-overlay"
                onClick={() => {
                  document.getElementById('mobile-job-type-dropdown')?.classList.add('hidden');
                  document.getElementById('mobile-dropdown-overlay')?.classList.add('hidden');
                }}
              ></div>
            </div>
            <div className="hidden md:block mx-4 bg-gray-200 dark:bg-gray-800 w-1 h-6"></div>
          </>
        )}

        <div className="pr-2">
          <Button
            className="text-gray-200 bg-gray-800 px-3 py-1 rounded-md whitespace-nowrap text-sm flex items-center justify-center h-[30px]"
            onClick={() => setShowAdvancedView(!showAdvancedView)}
          >
            <span className="hidden md:inline">{showAdvancedView ? 'Simple' : 'Advanced'}</span>
            <span className="md:hidden text-lg">
              {showAdvancedView ? <FaList /> : <FaSlidersH />}
            </span>
          </Button>
        </div>
        <div>
          <Button
            className="text-gray-200 bg-green-800 px-3 py-1 rounded-md whitespace-nowrap text-sm flex items-center justify-center h-[30px]"
            onClick={() => saveJob()}
            disabled={status === 'saving'}
          >
            <span className="hidden md:inline">{status === 'saving' ? 'Saving...' : runId ? 'Update' : 'Create'}</span>
            <span className="md:hidden text-lg">
              <FaSave />
            </span>
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
          <ErrorBoundary
            fallback={
              <div className="flex items-center justify-center h-64 text-lg text-red-600 font-medium bg-red-100 dark:bg-red-900/20 dark:text-red-400 border border-red-300 dark:border-red-700 rounded-lg">
                Advanced job detected. Please switch to advanced view to continue.
              </div>
            }
          >
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
