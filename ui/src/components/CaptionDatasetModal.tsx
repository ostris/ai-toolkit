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

export interface CaptionDatasetModalState {
  datasetPath: string;
  onClose?: () => void;
}

export const captionDatasetModalState = createGlobalState<CaptionDatasetModalState | null>(null);

export const openCaptionDatasetModal = (datasetPath: string, onClose?: () => void) => {
  captionDatasetModalState.set({ datasetPath, onClose });
};

export const CaptionDatasetModal: React.FC = () => {
  const [modalInfo, setModalInfo] = captionDatasetModalState.use();
  const [jobConfig, setJobConfig] = useNestedState<CaptionJobConfig>(objectCopy(defaultCaptionJobConfig));
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const [activeTab, setActiveTab] = useState<'simple' | 'advanced'>('simple');
  const open = modalInfo !== null;
  const isSavingRef = useRef(false);
  const showGPUSelect = !isMac();

  useFromNull(() => {
    // reset the state
    setJobConfig(objectCopy(defaultCaptionJobConfig));
    setActiveTab('simple');
    // set the path_to_caption
    if (modalInfo?.datasetPath) {
      setJobConfig(modalInfo.datasetPath, 'config.process[0].caption.path_to_caption');
    }
  }, [modalInfo]);

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
    setModalInfo(null);
  };

  const saveJob = async () => {
    if (isSavingRef.current) return;
    if (!modalInfo?.datasetPath) {
      alert('Dataset path is missing. Please try again.');
      return;
    }
    isSavingRef.current = true;

    apiClient
      .post('/api/jobs', {
        id: null,
        name: uuidv4(),
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
      <div className="space-y-4 text-gray-200">
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
