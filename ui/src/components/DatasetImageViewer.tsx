'use client';
import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { Cog } from 'lucide-react';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import classNames from 'classnames';
import { openConfirm } from './ConfirmModal';
import { apiClient } from '@/utils/api';
import { isVideo, isAudio } from '@/utils/basic';
import AudioPlayer from './AudioPlayer';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

interface Props {
  imgPath: string | null; // current image path
  imageList: string[]; // all dataset image paths
  onChange: (nextPath: string | null) => void; // parent setter
  refreshImages?: () => void;
  onCaptionSaved?: (imgPath: string, caption: string) => void;
}

export default function DatasetImageViewer({ imgPath, imageList, onChange, refreshImages, onCaptionSaved }: Props) {
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(Boolean(imgPath));
  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const [isCaptionLoaded, setIsCaptionLoaded] = useState<boolean>(false);
  const captionRef = useRef<string>('');
  const savedCaptionRef = useRef<string>('');
  const currentImgPathRef = useRef<string | null>(null);
  const captionAbortRef = useRef<AbortController | null>(null);

  useEffect(() => setMounted(true), []);

  // open/close based on external value
  useEffect(() => {
    setIsOpen(Boolean(imgPath));
  }, [imgPath]);

  // after close, clear parent state post-transition
  useEffect(() => {
    if (!isOpen && imgPath) {
      const t = setTimeout(() => onChange(null), 300);
      return () => clearTimeout(t);
    }
  }, [isOpen, imgPath, onChange]);

  // Keep refs in sync so we can save the latest caption on unmount/navigation
  useEffect(() => {
    captionRef.current = caption;
  }, [caption]);
  useEffect(() => {
    savedCaptionRef.current = savedCaption;
  }, [savedCaption]);

  const filename = useMemo(() => {
    if (!imgPath) return '';
    if (imgPath.includes('\\')) {
      const parts = imgPath.split('\\');
      return parts[parts.length - 1];
    }
    return imgPath.split('/').pop() || '';
  }, [imgPath]);

  const currentIndex = useMemo(() => {
    if (!imgPath) return -1;
    return imageList.findIndex(img => img === imgPath);
  }, [imgPath, imageList]);

  const saveCaptionForPath = useCallback(
    (path: string, value: string, prevSaved: string) => {
      const trimmed = value.trim();
      if (trimmed === prevSaved.trim()) return;
      apiClient
        .post('/api/img/caption', { imgPath: path, caption: trimmed })
        .then(() => {
          if (currentImgPathRef.current === path) {
            setSavedCaption(trimmed);
          }
          onCaptionSaved?.(path, trimmed);
        })
        .catch(error => {
          console.error('Error saving caption:', error);
        });
    },
    [onCaptionSaved],
  );

  const saveCaption = useCallback(() => {
    if (!imgPath) return;
    saveCaptionForPath(imgPath, caption, savedCaption);
  }, [imgPath, caption, savedCaption, saveCaptionForPath]);

  // Fetch caption whenever the image changes; save any pending edits on the previous image first
  useEffect(() => {
    const previousPath = currentImgPathRef.current;
    if (previousPath && previousPath !== imgPath) {
      saveCaptionForPath(previousPath, captionRef.current, savedCaptionRef.current);
    }
    currentImgPathRef.current = imgPath;

    if (!imgPath) {
      setCaption('');
      setSavedCaption('');
      setIsCaptionLoaded(false);
      return;
    }

    captionAbortRef.current?.abort();
    const controller = new AbortController();
    captionAbortRef.current = controller;
    setIsCaptionLoaded(false);
    setCaption('');
    setSavedCaption('');

    apiClient
      .post('/api/caption/get', { imgPath }, { signal: controller.signal })
      .then(res => res.data)
      .then(data => {
        if (controller.signal.aborted) return;
        const text = data ? `${data}` : '';
        setCaption(text);
        setSavedCaption(text);
        setIsCaptionLoaded(true);
      })
      .catch(error => {
        if (controller.signal.aborted) return;
        console.error('Error fetching caption:', error);
        setIsCaptionLoaded(true);
      });

    return () => {
      controller.abort();
    };
  }, [imgPath, saveCaptionForPath]);

  // Save any pending caption when the viewer fully unmounts
  useEffect(() => {
    return () => {
      const path = currentImgPathRef.current;
      if (path) {
        saveCaptionForPath(path, captionRef.current, savedCaptionRef.current);
      }
    };
  }, [saveCaptionForPath]);

  const onCancel = useCallback(() => {
    saveCaption();
    setIsOpen(false);
  }, [saveCaption]);

  const setImageAtIndex = useCallback(
    (idx: number) => {
      if (idx < 0 || idx >= imageList.length) return;
      onChange(imageList[idx]);
    },
    [imageList, onChange],
  );

  const handlePrev = useCallback(() => {
    if (currentIndex <= 0) return;
    setImageAtIndex(currentIndex - 1);
  }, [currentIndex, setImageAtIndex]);

  const handleNext = useCallback(() => {
    if (currentIndex === -1 || currentIndex >= imageList.length - 1) return;
    setImageAtIndex(currentIndex + 1);
  }, [currentIndex, imageList.length, setImageAtIndex]);

  const handleDelete = useCallback(() => {
    if (!imgPath) return;
    openConfirm({
      title: 'Delete Image',
      message: `Are you sure you want to delete this image? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/img/delete', { imgPath })
          .then(() => {
            console.log('Image deleted:', imgPath);
            onChange(null);
            if (refreshImages) refreshImages();
          })
          .catch(error => {
            console.error('Error deleting image:', error);
          });
      },
    });
  }, [imgPath, onChange, refreshImages]);

  // keyboard events while open — skip nav while caption textarea is focused
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen) return;
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      const isTyping = tag === 'TEXTAREA' || tag === 'INPUT' || (target?.isContentEditable ?? false);

      if (event.key === 'Escape') {
        onCancel();
        return;
      }
      if (isTyping) return;

      switch (event.key) {
        case 'ArrowLeft':
        case 'ArrowUp':
          handlePrev();
          break;
        case 'ArrowRight':
        case 'ArrowDown':
          handleNext();
          break;
        case 'Delete':
        case 'Backspace':
          handleDelete();
          break;
        default:
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel, handlePrev, handleNext, handleDelete]);

  // Touch swipe navigation
  const touchStartRef = useRef<{ x: number; y: number } | null>(null);
  const multiTouchRef = useRef(false);
  const zoomedRef = useRef(false);
  const SWIPE_THRESHOLD = 40;

  const onTouchStart = useCallback((e: React.TouchEvent) => {
    if (e.touches.length > 1) {
      multiTouchRef.current = true;
      touchStartRef.current = null;
      return;
    }
    multiTouchRef.current = false;
    const t = e.touches[0];
    if (!t) return;
    touchStartRef.current = { x: t.clientX, y: t.clientY };
  }, []);

  const onTouchEnd = useCallback(
    (e: React.TouchEvent) => {
      const start = touchStartRef.current;
      touchStartRef.current = null;
      if (multiTouchRef.current || zoomedRef.current) {
        if (e.touches.length === 0) multiTouchRef.current = false;
        return;
      }
      if (!start) return;
      const t = e.changedTouches[0];
      if (!t) return;
      const dx = t.clientX - start.x;
      const dy = t.clientY - start.y;
      const absX = Math.abs(dx);
      const absY = Math.abs(dy);
      if (absX < SWIPE_THRESHOLD && absY < SWIPE_THRESHOLD) return;
      if (absX > absY) {
        if (dx < 0) handleNext();
        else handlePrev();
      } else {
        if (dy < 0) handleNext();
        else handlePrev();
      }
    },
    [handleNext, handlePrev],
  );

  const handleCaptionKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
  };

  const isCaptionCurrent = caption.trim() === savedCaption.trim();

  if (!mounted) return null;

  return createPortal(
    <Dialog open={isOpen} onClose={onCancel} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-0 sm:p-4 text-center">
          <DialogPanel
            transition
            onTouchStart={onTouchStart}
            onTouchEnd={onTouchEnd}
            className="relative transform rounded-none sm:rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in w-full sm:w-auto sm:max-w-[95%] sm:max-h-[95vh] data-closed:sm:translate-y-0 data-closed:sm:scale-95 flex flex-col overflow-hidden touch-pan-y"
          >
            <div className="overflow-hidden flex items-center justify-center">
              {imgPath &&
                (isAudio(imgPath) ? (
                  <div className="w-[500px] h-[500px] max-w-full sm:max-w-[95vw] max-h-[70vh]">
                    <AudioPlayer src={`/api/img/${encodeURIComponent(imgPath)}`} title={filename} autoPlay />
                  </div>
                ) : isVideo(imgPath) ? (
                  <video
                    src={`/api/img/${encodeURIComponent(imgPath)}`}
                    className="w-auto h-auto max-w-full sm:max-w-[95vw] max-h-[70vh] object-contain"
                    preload="none"
                    playsInline
                    loop
                    autoPlay
                    controls={true}
                  />
                ) : (
                  <TransformWrapper
                    key={imgPath}
                    initialScale={1}
                    minScale={1}
                    maxScale={6}
                    doubleClick={{ mode: 'toggle', step: 2 }}
                    wheel={{ step: 0.2 }}
                    panning={{ disabled: false, allowRightClickPan: false }}
                    onTransform={(_ref, state) => {
                      zoomedRef.current = state.scale > 1.01;
                    }}
                  >
                    <TransformComponent>
                      <img
                        src={`/api/img/${encodeURIComponent(imgPath)}`}
                        alt="Dataset Image"
                        draggable={false}
                        className="w-auto h-auto max-w-full sm:max-w-[95vw] max-h-[70vh] object-contain select-none !pointer-events-auto"
                      />
                    </TransformComponent>
                  </TransformWrapper>
                ))}
            </div>
            <div className="bg-gray-950 text-sm flex flex-col px-4 py-2 gap-2">
              <div className="flex items-center justify-between gap-4">
                <div className="text-xs text-gray-400 truncate min-w-0">
                  <span className="text-gray-500 mr-1">File:</span>
                  <span className="text-gray-300">{filename}</span>
                </div>
                <div className="text-xs text-gray-400 whitespace-nowrap">
                  {currentIndex >= 0 ? `${currentIndex + 1} / ${imageList.length}` : ''}
                </div>
              </div>
              <div
                className={classNames('rounded border-2 bg-gray-900 transition-colors', {
                  'border-blue-500': !isCaptionCurrent,
                  'border-gray-700': isCaptionCurrent,
                })}
              >
                <textarea
                  className="w-full bg-transparent text-gray-100 text-sm p-2 resize-none outline-none focus:ring-0 focus:outline-none"
                  placeholder={isCaptionLoaded ? 'Add a caption...' : 'Loading caption...'}
                  value={caption}
                  rows={3}
                  onChange={e => setCaption(e.target.value)}
                  onKeyDown={handleCaptionKeyDown}
                  onBlur={saveCaption}
                  disabled={!isCaptionLoaded}
                />
              </div>
            </div>
            <div className="absolute top-2 right-2 bg-gray-900 rounded-full p-1 leading-[0px] opacity-50 hover:opacity-100 z-20">
              <Menu>
                <MenuButton>
                  <Cog />
                </MenuButton>
                <MenuItems
                  anchor="bottom end"
                  className="bg-gray-900 border border-gray-700 rounded shadow-lg w-48 px-2 py-2 mt-1 z-50"
                >
                  {imgPath && isAudio(imgPath) && (
                    <MenuItem>
                      <a
                        className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded block"
                        href={`/api/img/${encodeURIComponent(imgPath)}`}
                        download={filename}
                      >
                        Download
                      </a>
                    </MenuItem>
                  )}
                  <MenuItem>
                    <div className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded" onClick={handleDelete}>
                      Delete Image
                    </div>
                  </MenuItem>
                </MenuItems>
              </Menu>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
}
