'use client';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaUpload } from 'react-icons/fa';
import { useEffect, useRef } from 'react';

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
  const open = addImagesModalInfo !== null;
  const panelRef = useRef<HTMLDivElement>(null);

  const onCancel = () => {
    setAddImagesModalInfo(null);
  };

  const onDone = () => {
    if (addImagesModalInfo?.onComplete) {
      addImagesModalInfo.onComplete();
    }
    setAddImagesModalInfo(null);
  };

  // Close modal as soon as files are dragged in so the FullscreenDropOverlay can handle the drop
  useEffect(() => {
    if (!open) return;

    const handleDragEnter = (e: DragEvent) => {
      const types = e?.dataTransfer?.types;
      if (types && Array.from(types).includes('Files')) {
        setAddImagesModalInfo(null);
      }
    };

    window.addEventListener('dragenter', handleDragEnter);
    return () => window.removeEventListener('dragenter', handleDragEnter);
  }, [open, setAddImagesModalInfo]);

  return (
    <Dialog open={open} onClose={onCancel} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />

      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
          <DialogPanel
            ref={panelRef}
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
                    className="h-40 w-full flex flex-col items-center justify-center border-2 border-dashed rounded-lg border-gray-600"
                  >
                    <FaUpload className="size-8 mb-3 text-gray-400" />
                    <p className="text-sm text-gray-200 text-center">
                      Drag & drop files anywhere on the page to upload
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
              <button
                type="button"
                onClick={onDone}
                className="inline-flex w-full justify-center rounded-md bg-slate-600 px-3 py-2 text-sm font-semibold text-white shadow-xs sm:ml-3 sm:w-auto"
              >
                Done
              </button>
              <button
                type="button"
                data-autofocus
                onClick={onCancel}
                className="mt-3 inline-flex w-full justify-center rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-800 sm:mt-0 sm:w-auto ring-0"
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
