'use client';
import { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { isVideo } from '@/utils/basic';
import { apiClient } from '@/utils/api';

interface Props {
  imgPath: string | null;
  images: string[];
  onChange: (nextPath: string | null) => void;
  apiBase?: string;
}

export default function DatasetImageViewer({ imgPath, images, onChange, apiBase = '/api/img' }: Props) {
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(Boolean(imgPath));
  const [caption, setCaption] = useState<string>('');
  const [captionLoading, setCaptionLoading] = useState(false);
  const captionRequestRef = useRef(0);

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

  useEffect(() => {
    if (!imgPath) {
      setCaption('');
      return;
    }
    const requestId = ++captionRequestRef.current;
    setCaptionLoading(true);
    apiClient
      .post('/api/caption/get', { imgPath })
      .then(res => {
        if (requestId !== captionRequestRef.current) return;
        const data = res.data;
        setCaption(data ? `${data}` : '');
      })
      .catch(() => {
        if (requestId !== captionRequestRef.current) return;
        setCaption('');
      })
      .finally(() => {
        if (requestId === captionRequestRef.current) setCaptionLoading(false);
      });
  }, [imgPath]);

  if (!mounted) return null;

  return createPortal(
    <Dialog open={isOpen} onClose={onCancel} className="relative z-50">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div className="fixed inset-0 z-50 w-screen overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4 text-center">
          <DialogPanel
            transition
            className="relative transform rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in max-w-[95%] max-h-[95vh] data-closed:sm:translate-y-0 data-closed:sm:scale-95 flex flex-col overflow-hidden"
          >
            <div className="overflow-hidden flex items-center justify-center" style={{ maxHeight: '75vh' }}>
              {imgPath &&
                (isVideo(imgPath) ? (
                  <video
                    src={`${apiBase}/${encodeURIComponent(imgPath)}`}
                    className="w-auto h-auto max-w-[95vw] max-h-[75vh] object-contain"
                    preload="none"
                    playsInline
                    loop
                    autoPlay
                    controls={true}
                  />
                ) : (
                  <img
                    src={`${apiBase}/${encodeURIComponent(imgPath)}`}
                    alt="Dataset Image"
                    className="w-auto h-auto max-w-[95vw] max-h-[75vh] object-contain"
                  />
                ))}
            </div>
            <div className="bg-gray-900 border-t border-gray-700 px-4 py-3 overflow-y-auto" style={{ height: '15vh' }}>
              {captionLoading ? (
                <p className="text-gray-400 text-sm">Loading caption...</p>
              ) : caption ? (
                <p className="text-gray-200 text-sm whitespace-pre-wrap">{caption}</p>
              ) : (
                <p className="text-gray-500 text-sm italic">No caption</p>
              )}
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
