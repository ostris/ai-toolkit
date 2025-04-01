'use client';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaUpload } from 'react-icons/fa';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { apiClient } from '@/utils/api';

export interface AddImagesModalState {
  datasetName: string;
  onComplete?: () => void;
}

export const addImagesModalState = createGlobalState<AddImagesModalState | null>(null);

export const openImagesModal = (datasetName: string, onComplete: () => void) => {
  addImagesModalState.set({ datasetName, onComplete });
};

export default function AddImagesModal() {
  const [addImagesModalInfo, setAddImagesModalInfo] = addImagesModalState.use();
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const open = addImagesModalInfo !== null;

  const onCancel = () => {
    if (!isUploading) {
      setAddImagesModalInfo(null);
    }
  };

  const onDone = () => {
    if (addImagesModalInfo?.onComplete && !isUploading) {
      addImagesModalInfo.onComplete();
      setAddImagesModalInfo(null);
    }
  };

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      setIsUploading(true);
      setUploadProgress(0);
      
      const batchSize = 1000;
      const batches: File[][] = [];
      
      for (let i = 0; i < acceptedFiles.length; i += batchSize) {
        batches.push(acceptedFiles.slice(i, i + batchSize));
      }
      
      const totalFiles = acceptedFiles.length;
      let filesUploaded = 0;
      
      try {
        for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
          const batch = batches[batchIndex];
          const formData = new FormData();
          
          batch.forEach(file => {
            formData.append('files', file);
          });
          
          formData.append('datasetName', addImagesModalInfo?.datasetName || '');
          
          const baseProgress = (filesUploaded / totalFiles) * 100;
          const batchWeight = batch.length / totalFiles;
          
          await apiClient.post(`/api/datasets/upload`, formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: progressEvent => {
              const batchProgress = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 100));
              const currentBatchContribution = (batchProgress / 100) * batchWeight * 100;
              const overallProgress = Math.round(baseProgress + currentBatchContribution);
              setUploadProgress(overallProgress);
            },
            timeout: 0, // Disable timeout
          });
          filesUploaded += batch.length;
        }
        
        onDone();
      } catch (error) {
        console.error('Upload failed:', error);
      } finally {
        setIsUploading(false);
        setUploadProgress(0);
      }
    },
    [addImagesModalInfo],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'],
      'text/*': ['.txt'],
    },
    multiple: true,
  });

  return (
    <Dialog open={open} onClose={onCancel} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />

      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
          <DialogPanel
            transition
            className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-lg data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="bg-gray-800 px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <div className="text-center">
                <DialogTitle as="h3" className="text-base font-semibold text-gray-200 mb-4">
                  Add Images to: {addImagesModalInfo?.datasetName}
                </DialogTitle>
                <div className="w-full">
                  <div
                    {...getRootProps()}
                    className={`h-40 w-full flex flex-col items-center justify-center border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-200
                      ${isDragActive ? 'border-blue-500 bg-blue-50/10' : 'border-gray-600'}`}
                  >
                    <input {...getInputProps()} />
                    <FaUpload className="size-8 mb-3 text-gray-400" />
                    <p className="text-sm text-gray-200 text-center">
                      {isDragActive ? 'Drop the files here...' : 'Drag & drop files here, or click to select files'}
                    </p>
                  </div>
                  {isUploading && (
                    <div className="mt-4">
                      <div className="w-full bg-gray-700 rounded-full h-2.5">
                        <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${uploadProgress}%` }}></div>
                      </div>
                      <p className="text-sm text-gray-300 mt-2 text-center">Uploading... {uploadProgress}%</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
              <button
                type="button"
                onClick={onDone}
                disabled={isUploading}
                className={`inline-flex w-full justify-center rounded-md bg-slate-600 px-3 py-2 text-sm font-semibold text-white shadow-xs sm:ml-3 sm:w-auto
                  ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                Done
              </button>
              <button
                type="button"
                data-autofocus
                onClick={onCancel}
                disabled={isUploading}
                className={`mt-3 inline-flex w-full justify-center rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-800 sm:mt-0 sm:w-auto ring-0
                  ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                Cancel
              </button>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}
