'use client';
import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { Cog, SquareDashed, Pencil } from 'lucide-react';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import classNames from 'classnames';
import { openConfirm } from './ConfirmModal';
import { apiClient } from '@/utils/api';
import { isVideo, isAudio } from '@/utils/basic';
import AudioPlayer from './AudioPlayer';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import BoundingBoxOverlay, { BoundingBoxEditor, parseBoundingBoxes, extractBoxes } from './BoundingBoxOverlay';

function safeParse(text: string): any {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

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
  const [showBoxes, setShowBoxes] = useState<boolean>(false);
  const [isEditingBoxes, setIsEditingBoxes] = useState<boolean>(false);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const captionRef = useRef<string>('');
  const savedCaptionRef = useRef<string>('');
  const currentImgPathRef = useRef<string | null>(null);
  const captionAbortRef = useRef<AbortController | null>(null);

  useEffect(() => setMounted(true), []);

  // Leave box-edit mode whenever the image changes, to avoid accidental edits.
  useEffect(() => {
    setIsEditingBoxes(false);
    setSelectedBoxIndex(null);
    setIsDrawing(false);
  }, [imgPath]);

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

  // Stable handle to the latest saveCaptionForPath so the fetch effect doesn't
  // re-run (and re-fetch, blanking the caption) every time a save changes its
  // identity via the parent's onCaptionSaved.
  const saveCaptionForPathRef = useRef(saveCaptionForPath);
  useEffect(() => {
    saveCaptionForPathRef.current = saveCaptionForPath;
  }, [saveCaptionForPath]);

  const saveCaption = useCallback(() => {
    if (!imgPath) return;
    saveCaptionForPath(imgPath, caption, savedCaption);
  }, [imgPath, caption, savedCaption, saveCaptionForPath]);

  // Fetch caption whenever the image changes; save any pending edits on the previous image first
  useEffect(() => {
    const previousPath = currentImgPathRef.current;
    if (previousPath && previousPath !== imgPath) {
      saveCaptionForPathRef.current(previousPath, captionRef.current, savedCaptionRef.current);
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
      // transformResponse identity: keep the caption as a raw string. Axios's
      // default parses any JSON-looking body into an object (our bbox captions
      // are JSON), which would render as "[object Object]".
      .post('/api/caption/get', { imgPath }, { signal: controller.signal, transformResponse: [d => d] })
      .then(res => res.data)
      .then(data => {
        if (controller.signal.aborted) return;
        const text = typeof data === 'string' ? data : data ? `${data}` : '';
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
  }, [imgPath]);

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

  // Mutate the caption JSON's element array, updating the textarea ONLY (no
  // network). Persisting happens via the Save button / nav auto-save. Returns
  // whatever the mutator returns.
  const editCaption = useCallback(
    (fn: (elements: any[], data: any) => any): any => {
      let data: any;
      try {
        data = JSON.parse(caption);
      } catch {
        return undefined;
      }
      const elements = data?.compositional_deconstruction?.elements;
      if (!Array.isArray(elements)) return undefined;
      const result = fn(elements, data);
      setCaption(JSON.stringify(data, null, 2));
      return result;
    },
    [caption],
  );

  const handleBoxChange = useCallback(
    (elementIndex: number, box: { y1: number; x1: number; y2: number; x2: number }) => {
      editCaption(els => {
        if (els[elementIndex]) els[elementIndex].bbox = [box.y1, box.x1, box.y2, box.x2];
      });
    },
    [editCaption],
  );

  const handleDeleteBox = useCallback(
    (elementIndex: number) => {
      editCaption(els => {
        els.splice(elementIndex, 1);
      });
      setSelectedBoxIndex(null);
    },
    [editCaption],
  );

  const handleCreateBox = useCallback(
    (box: { y1: number; x1: number; y2: number; x2: number }) => {
      const newIndex = editCaption(els => {
        els.push({ type: 'obj', bbox: [box.y1, box.x1, box.y2, box.x2], desc: '' });
        return els.length - 1;
      });
      setSelectedBoxIndex(typeof newIndex === 'number' ? newIndex : null);
      setIsDrawing(false);
    },
    [editCaption],
  );

  const handleFieldChange = useCallback(
    (field: 'desc' | 'text', value: string) => {
      editCaption(els => {
        if (selectedBoxIndex != null && els[selectedBoxIndex]) els[selectedBoxIndex][field] = value;
      });
    },
    [editCaption, selectedBoxIndex],
  );

  const handleTypeChange = useCallback(
    (type: 'obj' | 'text') => {
      editCaption(els => {
        const el = selectedBoxIndex != null ? els[selectedBoxIndex] : null;
        if (!el) return;
        el.type = type;
        if (type === 'text') {
          if (el.text == null) el.text = '';
        } else {
          delete el.text;
        }
      });
    },
    [editCaption, selectedBoxIndex],
  );

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
          // While editing boxes, Delete removes the selected box (never the image).
          if (isEditingBoxes) {
            if (selectedBoxIndex != null) handleDeleteBox(selectedBoxIndex);
          } else {
            handleDelete();
          }
          break;
        default:
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel, handlePrev, handleNext, handleDelete, isEditingBoxes, selectedBoxIndex, handleDeleteBox]);

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
  const boundingBoxes = useMemo(() => parseBoundingBoxes(caption), [caption]);
  const canShowBoxes = Boolean(boundingBoxes && imgPath && !isAudio(imgPath) && !isVideo(imgPath));

  // Boxes and the selected element are derived from the (locally edited) caption.
  const editBoxes = useMemo(() => extractBoxes(safeParse(caption)), [caption]);
  const selectedElement = useMemo(() => {
    if (selectedBoxIndex == null) return null;
    return safeParse(caption)?.compositional_deconstruction?.elements?.[selectedBoxIndex] ?? null;
  }, [caption, selectedBoxIndex]);

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
                    doubleClick={{ mode: 'toggle', step: 2, disabled: isEditingBoxes }}
                    wheel={{ step: 0.2, disabled: isEditingBoxes }}
                    panning={{ disabled: isEditingBoxes, allowRightClickPan: false }}
                    onTransform={(_ref, state) => {
                      zoomedRef.current = state.scale > 1.01;
                    }}
                  >
                    <TransformComponent>
                      <div className="relative">
                        <img
                          src={`/api/img/${encodeURIComponent(imgPath)}`}
                          alt="Dataset Image"
                          draggable={false}
                          className="w-auto h-auto max-w-full sm:max-w-[95vw] max-h-[70vh] object-contain select-none !pointer-events-auto"
                        />
                        {isEditingBoxes ? (
                          <BoundingBoxEditor
                            boxes={editBoxes}
                            selectedIndex={selectedBoxIndex}
                            drawing={isDrawing}
                            onSelect={setSelectedBoxIndex}
                            onChangeBox={handleBoxChange}
                            onCreateBox={handleCreateBox}
                          />
                        ) : (
                          showBoxes && boundingBoxes && <BoundingBoxOverlay boxes={boundingBoxes} />
                        )}
                      </div>
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
              {isEditingBoxes && (
                <div className="rounded border border-gray-700 bg-gray-900 p-2 flex flex-col gap-2 text-xs">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setIsDrawing(d => !d)}
                      className={classNames('px-2 py-1 rounded border', {
                        'bg-blue-600 border-blue-500 text-white': isDrawing,
                        'border-gray-600 text-gray-300 hover:bg-gray-800': !isDrawing,
                      })}
                    >
                      {isDrawing ? 'Cancel' : '+ Add Box'}
                    </button>
                    <span className="text-gray-500">
                      {isDrawing
                        ? 'Drag on the image to draw a new box'
                        : 'Click a box to select; drag to move, handles to resize'}
                    </span>
                    <button
                      type="button"
                      onClick={saveCaption}
                      disabled={isCaptionCurrent}
                      className={classNames('ml-auto px-3 py-1 rounded border', {
                        'bg-green-600 border-green-500 text-white hover:bg-green-500': !isCaptionCurrent,
                        'border-gray-700 text-gray-500 cursor-default': isCaptionCurrent,
                      })}
                    >
                      {isCaptionCurrent ? 'Saved' : 'Save'}
                    </button>
                  </div>
                  {selectedElement && (
                    <div className="flex flex-col gap-2 border-t border-gray-700 pt-2">
                      <div className="flex items-center gap-2">
                        <span className="text-gray-400">Type:</span>
                        <button
                          type="button"
                          onClick={() => handleTypeChange('obj')}
                          className={classNames('px-2 py-0.5 rounded border', {
                            'bg-cyan-600 border-cyan-500 text-white': selectedElement.type !== 'text',
                            'border-gray-600 text-gray-300 hover:bg-gray-800': selectedElement.type === 'text',
                          })}
                        >
                          Object
                        </button>
                        <button
                          type="button"
                          onClick={() => handleTypeChange('text')}
                          className={classNames('px-2 py-0.5 rounded border', {
                            'bg-amber-600 border-amber-500 text-white': selectedElement.type === 'text',
                            'border-gray-600 text-gray-300 hover:bg-gray-800': selectedElement.type !== 'text',
                          })}
                        >
                          Text
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeleteBox(selectedBoxIndex!)}
                          className="ml-auto px-2 py-0.5 rounded border border-red-700 text-red-400 hover:bg-red-900/40"
                        >
                          Delete
                        </button>
                      </div>
                      {selectedElement.type === 'text' && (
                        <label className="flex flex-col gap-1">
                          <span className="text-gray-400">Text (shown in image)</span>
                          <textarea
                            className="w-full bg-gray-950 text-gray-100 rounded border border-gray-700 p-1 resize-none outline-none focus:border-blue-500"
                            rows={2}
                            value={selectedElement.text ?? ''}
                            onChange={e => handleFieldChange('text', e.target.value)}
                          />
                        </label>
                      )}
                      <label className="flex flex-col gap-1">
                        <span className="text-gray-400">Description</span>
                        <textarea
                          className="w-full bg-gray-950 text-gray-100 rounded border border-gray-700 p-1 resize-none outline-none focus:border-blue-500"
                          rows={2}
                          value={selectedElement.desc ?? ''}
                          onChange={e => handleFieldChange('desc', e.target.value)}
                        />
                      </label>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="absolute top-2 right-2 flex items-center gap-2 z-20">
              {canShowBoxes && (
                <button
                  type="button"
                  onClick={() => {
                    const next = !showBoxes;
                    setShowBoxes(next);
                    if (!next) {
                      setIsEditingBoxes(false);
                      setSelectedBoxIndex(null);
                      setIsDrawing(false);
                    }
                  }}
                  title={showBoxes ? 'Hide bounding boxes' : 'Show bounding boxes'}
                  className={classNames('bg-gray-900 rounded-full p-1 leading-[0px] hover:opacity-100', {
                    'opacity-100 text-blue-400': showBoxes,
                    'opacity-50': !showBoxes,
                  })}
                >
                  <SquareDashed />
                </button>
              )}
              {((canShowBoxes && showBoxes) || isEditingBoxes) && (
                <button
                  type="button"
                  onClick={() => {
                    const next = !isEditingBoxes;
                    setIsEditingBoxes(next);
                    if (!next) {
                      setSelectedBoxIndex(null);
                      setIsDrawing(false);
                    }
                  }}
                  title={isEditingBoxes ? 'Done editing boxes' : 'Edit bounding boxes'}
                  className={classNames('bg-gray-900 rounded-full p-1 leading-[0px] hover:opacity-100', {
                    'opacity-100 text-blue-400': isEditingBoxes,
                    'opacity-50': !isEditingBoxes,
                  })}
                >
                  <Pencil />
                </button>
              )}
              <div className="bg-gray-900 rounded-full p-1 leading-[0px] opacity-50 hover:opacity-100">
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
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
}
