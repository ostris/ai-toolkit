import React, { useState, useEffect } from 'react';
import { Modal } from '@/components/Modal';
import { createGlobalState } from 'react-global-hooks';
import { useFromNull } from '@/hooks/useFromNull';
import { SelectInput } from '@/components/formInputs';
import { CaptionJobConfig } from '@/types';
import { defaultCaptionJobConfig, handleCaptionerTypeChange } from '@/helpers/captionJobConfig';
import { objectCopy } from '@/utils/basic';
import { useNestedState } from '@/utils/hooks';
import { groupedCaptionerTypes } from '@/helpers/captionOptions';

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
  const open = modalInfo !== null;

  useFromNull(() => {
    // reset the state
    setJobConfig(objectCopy(defaultCaptionJobConfig));
  }, [modalInfo]);

  const handleClose = () => {
    if (modalInfo?.onClose) {
      modalInfo.onClose();
    }
    setModalInfo(null);
  };

  const handleCaptionDataset = () => {};

  return (
    <Modal isOpen={open} onClose={handleClose} title="Caption Dataset" size="lg">
      <div className="space-y-4 text-gray-200">
        <form
          onSubmit={e => {
            e.preventDefault();
            handleCaptionDataset();
          }}
        >
          <div className="text-sm text-gray-400">
            <SelectInput
              label="Captioner Type"
              value={jobConfig.config.process[0].type}
              onChange={value => {
                handleCaptionerTypeChange(jobConfig.config.process[0].type, value, jobConfig, setJobConfig);
              }}
              options={groupedCaptionerTypes}
            />
          </div>
          <div className="mt-4">
            {/* <TextInput label="Dataset Name" value={newDatasetName} onChange={value => setNewDatasetName(value)} /> */}
          </div>

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
