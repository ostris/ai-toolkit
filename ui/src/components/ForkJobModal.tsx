'use client';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaSpinner, FaTimes } from 'react-icons/fa';
import { useRef, useState } from 'react';
import { forkJob } from '@/utils/fork';

export interface ForkJobModalState {
  parentJobId: string;
  parentJobName: string;
  parentJobStep: number;
  onComplete?: (jobId: string, jobName: string) => void;
}

export const forkJobModalState = createGlobalState<ForkJobModalState | null>(null);

export const openForkJobModal = (
  parentJobId: string,
  parentJobName: string,
  parentJobStep: number,
  onComplete?: (jobId: string, jobName: string) => void,
) => {
  forkJobModalState.set({
    parentJobId,
    parentJobName,
    parentJobStep,
    onComplete,
  });
};

export default function ForkJobModal() {
  const [modalInfo, setModalInfo] = forkJobModalState.use();
  const open = modalInfo !== null;

  const [newJobName, setNewJobName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submitRef = useRef(false);

  const handleClose = () => {
    if (!isLoading) {
      setModalInfo(null);
      setNewJobName('');
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (submitRef.current || isLoading) return;
    submitRef.current = true;

    try {
      setIsLoading(true);
      setError(null);

      const jobNameToUse = newJobName.trim();

      if (!jobNameToUse) {
        setError('Job name is required');
        submitRef.current = false;
        return;
      }

      if (!modalInfo) {
        setError('Modal info missing');
        submitRef.current = false;
        return;
      }

      const result = await forkJob(modalInfo.parentJobId, {
        newJobName: jobNameToUse,
        copyCheckpoint: true, // Always copy checkpoint when forking
      });

      if (!result.success) {
        setError(result.error || 'Failed to fork job');
        submitRef.current = false;
        return;
      }

      // Success
      if (modalInfo.onComplete && result.jobId) {
        modalInfo.onComplete(result.jobId, jobNameToUse);
      }

      handleClose();
    } catch (err) {
      console.error('Error forking job:', err);
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
      submitRef.current = false;
    } finally {
      setIsLoading(false);
    }
  };

  if (!open || !modalInfo) return null;

  return (
    <Dialog open={open} as="div" className="relative z-50" onClose={handleClose}>
      <DialogBackdrop className="fixed inset-0 bg-black/50" />
      <div className="fixed inset-0 overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4">
          <DialogPanel className="relative w-full max-w-md transform rounded-lg bg-gray-800 text-left shadow-xl transition-all border border-gray-700">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-gray-700 px-6 py-4">
              <DialogTitle className="text-lg font-medium text-gray-100">
                Fork Job
              </DialogTitle>
              <button
                onClick={handleClose}
                disabled={isLoading}
                className="text-gray-400 hover:text-gray-200 disabled:opacity-50"
              >
                <FaTimes size={20} />
              </button>
            </div>

            {/* Content */}
            <form onSubmit={handleSubmit} className="p-6 space-y-4">
              {/* Parent Job Info */}
              <div className="bg-gray-900 rounded px-4 py-3 space-y-1">
                <p className="text-sm text-gray-400">Forking from</p>
                <p className="text-gray-100 font-medium">{modalInfo.parentJobName}</p>
                <p className="text-xs text-gray-500">Step: {modalInfo.parentJobStep}</p>
              </div>

              {/* Error Message */}
              {error && (
                <div className="bg-red-900/20 border border-red-700 rounded px-4 py-3 text-red-300 text-sm">
                  {error}
                </div>
              )}

              {/* Job Name Input */}
              <div>
                <label htmlFor="jobName" className="block text-sm font-medium text-gray-300 mb-2">
                  New Job Name
                </label>
                <input
                  id="jobName"
                  type="text"
                  value={newJobName}
                  onChange={(e) => setNewJobName(e.target.value)}
                  placeholder={`${modalInfo.parentJobName}_fork`}
                  disabled={isLoading}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Give your forked job a unique name
                </p>
              </div>

              {/* Copy Checkpoint Checkbox */}
              <div className="bg-blue-900/20 border border-blue-700 rounded px-4 py-3 text-blue-300 text-sm">
                ✓ Latest checkpoint will be automatically copied to continue training from step {modalInfo.parentJobStep}
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-4 border-t border-gray-700">
                <button
                  type="button"
                  onClick={handleClose}
                  disabled={isLoading}
                  className="flex-1 px-4 py-2 text-gray-300 bg-gray-700 hover:bg-gray-600 rounded font-medium transition disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium transition disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isLoading && <FaSpinner className="animate-spin" size={16} />}
                  {isLoading ? 'Forking...' : 'Fork Job'}
                </button>
              </div>
            </form>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}
