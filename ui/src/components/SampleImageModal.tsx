'use client';
import { useState, useEffect, useMemo } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';

export interface SampleImageModalState {
  imgPath: string;
  numSamples: number;
  sampleImages: string[];
}

export const sampleImageModalState = createGlobalState<SampleImageModalState | null>(null);

export const openSampleImage = (sampleImageProps: SampleImageModalState) => {
  sampleImageModalState.set(sampleImageProps);
};

export default function SampleImageModal() {
  const [imageModal, setImageModal] = sampleImageModalState.use();
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    if (imageModal) {
      setIsOpen(true);
    }
  }, [imageModal]);

  useEffect(() => {
    if (!isOpen) {
      // use timeout to allow the dialog to close before resetting the state
      setTimeout(() => {
        setImageModal(null);
      }, 500);
    }
  }, [isOpen]);

  const onCancel = () => {
    setIsOpen(false);
  };

  const imgInfo = useMemo(() => {
    const ii = {
      filename: '',
      step: 0,
      promptIdx: 0,
    };
    if (imageModal?.imgPath) {
      const filename = imageModal.imgPath.split('/').pop();
      if (!filename) return ii;
      // filename is <timestep>__<zero_pad_step>_<prompt_idx>.<ext>
      ii.filename = filename as string;
      const parts = filename
        .split('.')[0]
        .split('_')
        .filter(p => p !== '');
      if (parts.length === 3) {
        ii.step = parseInt(parts[1]);
        ii.promptIdx = parseInt(parts[2]);
      }
    }
    return ii;
  }, [imageModal]);

  const handleArrowUp = () => {
    if (!imageModal) return;
    console.log('Arrow Up pressed');
    // Change image to same sample but up one step
    const currentIdx = imageModal.sampleImages.findIndex(img => img === imageModal.imgPath);
    if (currentIdx === -1) return;
    const nextIdx = currentIdx - imageModal.numSamples;
    if (nextIdx < 0) return;
    openSampleImage({
      imgPath: imageModal.sampleImages[nextIdx],
      numSamples: imageModal.numSamples,
      sampleImages: imageModal.sampleImages,
    });
  };

  const handleArrowDown = () => {
    if (!imageModal) return;
    console.log('Arrow Down pressed');
    // Change image to same sample but down one step
    const currentIdx = imageModal.sampleImages.findIndex(img => img === imageModal.imgPath);
    if (currentIdx === -1) return;
    const nextIdx = currentIdx + imageModal.numSamples;
    if (nextIdx >= imageModal.sampleImages.length) return;
    openSampleImage({
      imgPath: imageModal.sampleImages[nextIdx],
      numSamples: imageModal.numSamples,
      sampleImages: imageModal.sampleImages,
    });
  };

  const handleArrowLeft = () => {
    if (!imageModal) return;
    if (imgInfo.promptIdx === 0) return;
    console.log('Arrow Left pressed');
    // go to previous sample
    const currentIdx = imageModal.sampleImages.findIndex(img => img === imageModal.imgPath);
    if (currentIdx === -1) return;
    const minIdx = currentIdx - imgInfo.promptIdx;
    const nextIdx = currentIdx - 1;
    if (nextIdx < minIdx) return;
    openSampleImage({
      imgPath: imageModal.sampleImages[nextIdx],
      numSamples: imageModal.numSamples,
      sampleImages: imageModal.sampleImages,
    });
  };

  const handleArrowRight = () => {
    if (!imageModal) return;
    console.log('Arrow Right pressed');
    // go to next sample
    const currentIdx = imageModal.sampleImages.findIndex(img => img === imageModal.imgPath);
    if (currentIdx === -1) return;
    const stepMinIdx = currentIdx - imgInfo.promptIdx;
    const maxIdx = stepMinIdx + imageModal.numSamples - 1;
    const nextIdx = currentIdx + 1;
    if (nextIdx > maxIdx) return;
    if (nextIdx >= imageModal.sampleImages.length) return;
    openSampleImage({
      imgPath: imageModal.sampleImages[nextIdx],
      numSamples: imageModal.numSamples,
      sampleImages: imageModal.sampleImages,
    });
  };

  // Handle keyboard events
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen) return;

      switch (event.key) {
        case 'Escape':
          onCancel();
          break;
        case 'ArrowUp':
          handleArrowUp();
          break;
        case 'ArrowDown':
          handleArrowDown();
          break;
        case 'ArrowLeft':
          handleArrowLeft();
          break;
        case 'ArrowRight':
          handleArrowRight();
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, imageModal, imgInfo]);

  return (
    <Dialog open={isOpen} onClose={onCancel} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />

      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4 text-center">
          <DialogPanel
            transition
            className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in max-w-[95%] max-h-[95vh] data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="flex justify-center items-center">
              {imageModal?.imgPath && (
                <img
                  src={`/api/img/${encodeURIComponent(imageModal.imgPath)}`}
                  alt="Sample Image"
                  className="max-w-full max-h-[calc(95vh-2rem)] object-contain"
                />
              )}
            </div>
            <div className="bg-gray-950 text-center text-sm p-2">step: {imgInfo.step}</div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}
