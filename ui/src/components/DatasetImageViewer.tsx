'use client';
import { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { isVideo } from '@/utils/basic';

interface Props {
  imgPath: string | null;
  images: string[];
  onChange: (nextPath: string | null) => void;
}

export default function DatasetImageViewer({ imgPath, images, onChange }: Props) {
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(Boolean(imgPath));

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    setIsOpen(Boolean(imgPath));
  }, [imgPath]);

  useEffect(() => {
    if (!isOpen && imgPath) {
      const t = setTimeout(() => onChange(null), 300);
      return () => clearTimeout(t);
    }
  }, [isOpen, imgPath, onChange]);

  const onCancel = useCallback(() => setIsOpen(false), []);

  const currentIndex = imgPath ? images.indexOf(imgPath) : -1;

  const handleArrowLeft = useCallback(() => {
    if (currentIndex <= 0) return;
    onChange(images[currentIndex - 1]);
  }, [currentIndex, images, onChange]);

  const handleArrowRight = useCallback(() => {
    if (currentIndex === -1 || currentIndex >= images.length - 1) return;
    onChange(images[currentIndex + 1]);
  }, [currentIndex, images, onChange]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen) return;
      switch (event.key) {
        case 'Escape':
          onCancel();
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
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel, handleArrowLeft, handleArrowRight]);

  if (!mounted) return null;

  return createPortal(
    <Dialog open={isOpen} onClose={onCancel} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4 text-center">
          <DialogPanel
            transition
            className="relative transform rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in max-w-[95%] max-h-[95vh] data-closed:sm:translate-y-0 data-closed:sm:scale-95 flex flex-col overflow-hidden"
          >
            <div className="overflow-hidden flex items-center justify-center">
              {imgPath &&
                (isVideo(imgPath) ? (
                  <video
                    src={`/api/img/${encodeURIComponent(imgPath)}`}
                    className="w-auto h-auto max-w-[95vw] max-h-[90vh] object-contain"
                    preload="none"
                    playsInline
                    loop
                    autoPlay
                    controls={true}
                  />
                ) : (
                  <img
                    src={`/api/img/${encodeURIComponent(imgPath)}`}
                    alt="Dataset Image"
                    className="w-auto h-auto max-w-[95vw] max-h-[90vh] object-contain"
                  />
                ))}
            </div>
            {images.length > 1 && (
              <div className="bg-gray-950 text-sm flex justify-between items-center px-4 py-2">
                <button
                  onClick={handleArrowLeft}
                  disabled={currentIndex <= 0}
                  className="text-gray-300 disabled:opacity-30 hover:text-white px-2"
                >
                  ← Prev
                </button>
                <span className="text-gray-400 text-xs">
                  {currentIndex + 1} / {images.length}
                </span>
                <button
                  onClick={handleArrowRight}
                  disabled={currentIndex >= images.length - 1}
                  className="text-gray-300 disabled:opacity-30 hover:text-white px-2"
                >
                  Next →
                </button>
              </div>
            )}
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
}
