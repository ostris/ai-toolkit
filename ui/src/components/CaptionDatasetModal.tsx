'use client';
import React, { useState, useEffect, useRef } from 'react';
import { Modal } from '@/components/Modal';
import { createGlobalState } from 'react-global-hooks';
import { useFromNull } from '@/hooks/useFromNull';
import { CaptionJobConfig } from '@/types';
import { defaultCaptionJobConfig } from '@/helpers/captionJobConfig';
import { objectCopy } from '@/utils/basic';
import { useNestedState } from '@/utils/hooks';
import { isMac } from '@/helpers/basic';
import useGPUInfo from '@/hooks/useGPUInfo';
import { apiClient } from '@/utils/api';
import { v4 as uuidv4 } from 'uuid';
import { startJob } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import CaptionSimpleJob from '@/components/CaptionSimpleJob';
import AdvancedConfigEditor from '@/components/AdvancedConfigEditor';
import { SelectInput } from '@/components/formInputs';
import { Loader2 } from 'lucide-react';

export interface CaptionDatasetModalState {
  datasetPath: string;
  jobId?: string | null;
  cloneId?: string | null;
  onClose?: () => void;
}

export const captionDatasetModalState = createGlobalState<CaptionDatasetModalState | null>(null);

export const openCaptionDatasetModal = (
  datasetPath: string,
  onClose?: () => void,
  options?: { jobId?: string | null; cloneId?: string | null },
) => {
  captionDatasetModalState.set({
    datasetPath,
    onClose,
    jobId: options?.jobId ?? null,
    cloneId: options?.cloneId ?? null,
  });
};

export const CaptionDatasetModal: React.FC = () => {
  const [modalInfo, setModalInfo] = captionDatasetModalState.use();
  const [jobConfig, setJobConfig] = useNestedState<CaptionJobConfig>(objectCopy(defaultCaptionJobConfig));
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [existingJobName, setExistingJobName] = useState<string | null>(null);
  const [hasLoadedExistingJob, setHasLoadedExistingJob] = useState(false);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const [activeTab, setActiveTab] = useState<'simple' | 'advanced'>('simple');
  const open = modalInfo !== null;
  const isSavingRef = useRef(false);
  const [isSaving, setIsSaving] = useState(false);
  const showGPUSelect = !isMac();
  const isLoadingExistingJob = !!(modalInfo?.jobId || modalInfo?.cloneId) && !hasLoadedExistingJob;
  const showLoadingOverlay = isLoadingExistingJob || isSaving;

  useFromNull(() => {
    // reset the state
    setJobConfig(objectCopy(defaultCaptionJobConfig));
    setActiveTab('simple');
    setExistingJobName(null);
    // set the path_to_caption
    if (modalInfo?.datasetPath) {
      setJobConfig(modalInfo.datasetPath, 'config.process[0].caption.path_to_caption');
    }
  }, [modalInfo]);

  // clone existing caption job
  useEffect(() => {
    if (modalInfo?.cloneId) {
      apiClient
        .get(`/api/jobs?id=${modalInfo.cloneId}`)
        .then(res => res.data)
        .then(data => {
          setGpuIDs(data.gpu_ids);
          const newJobConfig = JSON.parse(data.job_config);
          newJobConfig.config.name = `${newJobConfig.config.name}_copy`;
          setJobConfig(newJobConfig);
        })
        .catch(error => console.error('Error fetching caption job:', error))
        .finally(() => setHasLoadedExistingJob(true));
    }
  }, [modalInfo?.cloneId]);

  // load existing caption job for editing
  useEffect(() => {
    if (modalInfo?.jobId) {
      apiClient
        .get(`/api/jobs?id=${modalInfo.jobId}`)
        .then(res => res.data)
        .then(data => {
          setGpuIDs(data.gpu_ids);
          setExistingJobName(data.name);
          setJobConfig(JSON.parse(data.job_config));
        })
        .catch(error => console.error('Error fetching caption job:', error))
        .finally(() => setHasLoadedExistingJob(true));
    }
  }, [modalInfo?.jobId]);

  useEffect(() => {
    if (isGPUInfoLoaded) {
      if (gpuIDs === null && gpuList.length > 0) {
        setGpuIDs(`${gpuList[0].index}`);
      }
    }
  }, [gpuList, isGPUInfoLoaded]);

  const handleClose = () => {
    if (modalInfo?.onClose) {
      modalInfo.onClose();
    }
    setHasLoadedExistingJob(false);
    setModalInfo(null);
  };

  const saveJob = async () => {
    if (isSavingRef.current) return;
    if (!modalInfo?.datasetPath) {
      alert('Dataset path is missing. Please try again.');
      return;
    }
    isSavingRef.current = true;
    setIsSaving(true);

    const isEdit = !!modalInfo.jobId;

    apiClient
      .post('/api/jobs', {
        id: isEdit ? modalInfo.jobId : null,
        name: isEdit && existingJobName ? existingJobName : uuidv4(),
        gpu_ids: gpuIDs,
        job_config: jobConfig,
        job_type: 'caption',
        job_ref: modalInfo.datasetPath,
      })
      .then(async res => {
        const jobId = res.data.id;
        await startJob(jobId);
        // start the queue as well
        await startQueue(gpuIDs || '');
        isSavingRef.current = false;
        setIsSaving(false);
        handleClose();
      })
      .catch(error => {
        if (error.response?.status === 409) {
          alert('A caption job for this dataset already exists. Please check your jobs list.');
        } else {
          alert('Failed to save job. Please try again.');
        }
        console.log('Error saving training:', error);
        isSavingRef.current = false;
        setIsSaving(false);
      });
  };

  const tabButtonClass = (tab: 'simple' | 'advanced') =>
    `px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
      activeTab === tab
        ? 'border-blue-500 text-blue-400'
        : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-600'
    }`;

  return (
    <Modal isOpen={open} onClose={handleClose} title="Caption Dataset" size={activeTab === 'advanced' ? 'xl' : 'lg'}>
      <div className="relative space-y-4 text-gray-200">
        {showLoadingOverlay && (
          <div className="absolute -left-6 -right-6 -top-4 -bottom-4 z-10 flex items-center justify-center backdrop-blur-sm bg-gray-900/40">
            <Loader2 className="w-10 h-10 text-blue-400 animate-spin" />
          </div>
        )}
        <div className="flex items-center border-b border-gray-700 -mt-2">
          <button type="button" className={tabButtonClass('simple')} onClick={() => setActiveTab('simple')}>
            Simple
          </button>
          <button type="button" className={tabButtonClass('advanced')} onClick={() => setActiveTab('advanced')}>
            Advanced
          </button>
          <div className="flex-1" />
          {activeTab === 'advanced' && showGPUSelect && (
            <div className="pb-2">
              <SelectInput
                value={`${gpuIDs}`}
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            </div>
          )}
        </div>
        <form
          onSubmit={e => {
            e.preventDefault();
            saveJob();
          }}
        >
          {activeTab === 'simple' ? (
            <CaptionSimpleJob
              jobConfig={jobConfig}
              setJobConfig={setJobConfig}
              gpuIDs={gpuIDs}
              setGpuIDs={setGpuIDs}
              gpuList={gpuList}
              showGPUSelect={showGPUSelect}
            />
          ) : (
            <div className="h-[60vh] mt-2">
              <AdvancedConfigEditor config={jobConfig} setConfig={setJobConfig} />
            </div>
          )}

          <div className="mt-6 flex justify-end space-x-3">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
              onClick={handleClose}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              Add to Queue
            </button>
          </div>
        </form>
      </div>
    </Modal>
  );
};
