'use client';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaUpload } from 'react-icons/fa';
import { useCallback, useState } from 'react';
import { useDropzone, Accept } from 'react-dropzone';
import { apiClient } from '@/utils/api';

// State for checkpoint upload modal
interface CheckpointUploadState {
  // Full upload URL, typically `/api/jobs/${jobID}/checkpoints/upload`
  uploadUrl: string;
  // Called with the raw API response on success
  onComplete?: (response: any) => void;
  // Optional custom title
  title?: string;
}

const checkpointUploadState = createGlobalState<CheckpointUploadState | null>(null);

// Public helper to open the checkpoint upload modal
export function openCheckpointUploadModal(jobID: string, onComplete: (resp: any) => void, title = 'Import Checkpoint (.safetensors)') {
  checkpointUploadState.set({
    uploadUrl: `/api/jobs/${jobID}/checkpoints/upload`,
    onComplete,
    title,
  });
}

export default function CheckpointUploadModal() {
  const [info, setInfo] = checkpointUploadState.use();
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const open = info !== null;

  const onCancel = () => {
    if (!isUploading) setInfo(null);
  };

  const onDone = (resp: any) => {
    if (info?.onComplete) {
      info.onComplete(resp);
      setInfo(null);
    }
  };

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      setIsUploading(true);
      setUploadProgress(0);

      const formData = new FormData();
      // Checkpoint API expects a single field named 'file'
      formData.append('file', acceptedFiles[0]);

      try {
        const url = info?.uploadUrl as string;
        const resp = await apiClient.post(url, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: pe => {
            const percent = Math.round((pe.loaded * 100) / (pe.total || 100));
            setUploadProgress(percent);
          },
          timeout: 0,
        });
        onDone(resp.data);
      } catch (e) {
        console.error('Checkpoint upload failed:', e);
      } finally {
        setIsUploading(false);
        setUploadProgress(0);
      }
    },
    [info]
  );

  const accept: Accept = { 'application/octet-stream': ['.safetensors'] };
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept, multiple: false });

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
                  {info?.title || 'Import Checkpoint (.safetensors)'}
                </DialogTitle>
                <div className="w-full">
                  <div
                    {...getRootProps()}
                    className={`h-40 w-full flex flex-col items-center justify-center border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-200 ${
                      isDragActive ? 'border-blue-500 bg-blue-50/10' : 'border-gray-600'
                    }`}
                  >
                    <input {...getInputProps()} />
                    <FaUpload className="size-8 mb-3 text-gray-400" />
                    <p className="text-sm text-gray-200 text-center">
                      {isDragActive ? 'Drop the checkpoint here...' : 'Drag & drop a .safetensors file, or click to select one'}
                    </p>
                  </div>
                  {isUploading && (
                    <div className="mt-4">
                      <div className="w-full bg-gray-700 rounded-full h-2.5">
                        <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${uploadProgress}%` }} />
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
                data-autofocus
                onClick={onCancel}
                disabled={isUploading}
                className={`mt-3 inline-flex w-full justify-center rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-800 sm:mt-0 sm:w-auto ring-0 ${
                  isUploading ? 'opacity-50 cursor-not-allowed' : ''
                }`}
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
