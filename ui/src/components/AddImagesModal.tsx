'use client';
import { createGlobalState } from 'react-global-hooks';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaUpload, FaTimesCircle, FaSpinner } from 'react-icons/fa';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { apiClient } from '@/utils/api';

export interface AddImagesModalState {
  datasetName: string;
  onComplete?: () => void;
  openedByDrag?: boolean;
}

export const addImagesModalState = createGlobalState<AddImagesModalState | null>(null);

export const openImagesModal = (datasetName: string, onComplete: () => void) => {
  addImagesModalState.set({ datasetName, onComplete });
};

/** Call on a page that knows its datasetName — auto-opens the modal when files are dragged onto the page. */
export function useOpenImagesModalOnDrag(datasetName: string, onComplete: () => void) {
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    if (!datasetName) return;

    let depth = 0;
    const isFileDrag = (e: DragEvent) => {
      const types = e?.dataTransfer?.types;
      return !!types && Array.from(types).includes('Files');
    };

    const onDragEnter = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      depth += 1;
      if (depth === 1) {
        if (!addImagesModalState.get()) {
          addImagesModalState.set({ datasetName, onComplete: onCompleteRef.current, openedByDrag: true });
        }
      }
      e.preventDefault();
    };
    const onDragLeave = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) {
        const current = addImagesModalState.get();
        if (current?.openedByDrag) {
          addImagesModalState.set(null);
        }
      }
    };
    const onDrop = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      depth = 0;
      // Files were dropped — modal is now committed, no longer dismissable by drag-out
      const current = addImagesModalState.get();
      if (current?.openedByDrag) {
        addImagesModalState.set({ ...current, openedByDrag: false });
      }
    };

    window.addEventListener('dragenter', onDragEnter);
    window.addEventListener('dragleave', onDragLeave);
    window.addEventListener('drop', onDrop);
    return () => {
      window.removeEventListener('dragenter', onDragEnter);
      window.removeEventListener('dragleave', onDragLeave);
      window.removeEventListener('drop', onDrop);
    };
  }, [datasetName]);
}

type AcceptMap = { [mime: string]: string[] };
type FileStatus = 'pending' | 'uploading' | 'error';

interface FileEntry {
  id: number;
  file: File;
  status: FileStatus;
  progress: number;
  error?: string;
}

const MAX_CONCURRENT = 3;
const ROW_HEIGHT = 32;
const VISIBLE_ROWS = 8;

let nextId = 0;

export default function AddImagesModal() {
  const [modalInfo, setModalInfo] = addImagesModalState.use();
  const open = modalInfo !== null;

  const [isUploading, setIsUploading] = useState(false);
  const [fileEntries, setFileEntries] = useState<FileEntry[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [doneCount, setDoneCount] = useState(0);
  const [errorCount, setErrorCount] = useState(0);
  const abortRef = useRef(false);
  const modalInfoRef = useRef(modalInfo);
  modalInfoRef.current = modalInfo;

  const datasetName = modalInfo?.datasetName ?? '';

  const uploadSingleFile = useCallback(
    async (entry: FileEntry): Promise<'done' | 'error'> => {
      if (abortRef.current) return 'error';

      const id = entry.id;
      setFileEntries(prev =>
        prev.map(e => (e.id === id ? { ...e, status: 'uploading' as FileStatus, progress: 0 } : e)),
      );

      const formData = new FormData();
      formData.append('files', entry.file);
      formData.append('datasetName', datasetName || '');

      try {
        await apiClient.post('/api/datasets/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: pe => {
            const percent = Math.round(((pe.loaded || 0) * 100) / (pe.total || pe.loaded || 1));
            setFileEntries(prev => prev.map(e => (e.id === id ? { ...e, progress: percent } : e)));
          },
          timeout: 0,
        });
        setFileEntries(prev => prev.filter(e => e.id !== id));
        setDoneCount(prev => prev + 1);
        return 'done';
      } catch (err) {
        console.error(`Upload failed for ${entry.file.name}:`, err);
        setFileEntries(prev =>
          prev.map(e =>
            e.id === id
              ? { ...e, status: 'error' as FileStatus, error: err instanceof Error ? err.message : 'Upload failed' }
              : e,
          ),
        );
        setErrorCount(prev => prev + 1);
        return 'error';
      }
    },
    [datasetName],
  );

  const resetState = useCallback(() => {
    setFileEntries([]);
    setTotalCount(0);
    setDoneCount(0);
    setErrorCount(0);
  }, []);

  const processQueue = useCallback(
    async (entries: FileEntry[]) => {
      setIsUploading(true);
      abortRef.current = false;

      let nextIndex = 0;
      const runNext = async (): Promise<void> => {
        while (nextIndex < entries.length) {
          if (abortRef.current) return;
          const idx = nextIndex++;
          await uploadSingleFile(entries[idx]);
        }
      };

      const workers = Array.from({ length: Math.min(MAX_CONCURRENT, entries.length) }, () => runNext());
      await Promise.all(workers);

      setIsUploading(false);
      if (!abortRef.current) {
        modalInfoRef.current?.onComplete?.();
        setModalInfo(null);
        resetState();
      }
    },
    [uploadSingleFile, setModalInfo, resetState],
  );

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      const entries: FileEntry[] = acceptedFiles.map(file => ({
        id: nextId++,
        file,
        status: 'pending' as FileStatus,
        progress: 0,
      }));
      setFileEntries(entries);
      setTotalCount(entries.length);
      setDoneCount(0);
      setErrorCount(0);
      processQueue(entries);
    },
    [processQueue],
  );

  const handleCancel = useCallback(() => {
    abortRef.current = true;
    setIsUploading(false);
    setModalInfo(null);
    resetState();
  }, [setModalInfo, resetState]);

  const dropAccept = useMemo<AcceptMap>(
    () => ({
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'],
      'audio/*': ['.mp3', '.wav', '.flac', '.ogg'],
      'text/*': ['.txt'],
    }),
    [],
  );

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    open: openFilePicker,
  } = useDropzone({
    onDrop,
    accept: dropAccept,
    multiple: true,
    noClick: true,
    noKeyboard: true,
  });

  const overallPercent = totalCount > 0 ? Math.round(((doneCount + errorCount) / totalCount) * 100) : 0;

  return (
    <Dialog
      open={open}
      onClose={() => {
        if (!isUploading) handleCancel();
      }}
      className="relative z-10"
    >
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
                  Add Images to: {datasetName}
                </DialogTitle>

                {/* Drop zone + click to select */}
                <div {...getRootProps()} className="w-full">
                  <input {...getInputProps()} />
                  <div
                    onClick={() => {
                      if (!isUploading) openFilePicker();
                    }}
                    className={`h-40 w-full flex flex-col items-center justify-center border-2 border-dashed rounded-lg cursor-pointer transition-colors
                      ${isDragActive ? 'border-blue-400 bg-blue-500/10' : 'border-gray-600 hover:border-gray-400'}`}
                  >
                    <FaUpload className="size-8 mb-3 text-gray-400" />
                    {!isUploading ? (
                      <>
                        <p className="text-sm text-gray-200 text-center">Drag & drop files here or click to select</p>
                        <p className="text-xs text-gray-400 mt-1">Images, videos, or .txt supported</p>
                      </>
                    ) : (
                      <p className="text-sm text-gray-200 text-center">Drop more files to add to queue</p>
                    )}
                  </div>
                </div>

                {/* Upload progress */}
                {isUploading && (
                  <div className="mt-4">
                    <p className="text-sm font-semibold text-gray-200 mb-2">
                      Uploading… {doneCount + errorCount} / {totalCount}
                    </p>
                    <div className="w-full h-2.5 bg-white/20 rounded-full overflow-hidden">
                      <div
                        className="h-2.5 bg-blue-500 rounded-full transition-[width] duration-150 ease-linear"
                        style={{ width: `${overallPercent}%` }}
                      />
                    </div>
                    {errorCount > 0 && (
                      <p className="text-xs text-red-400 mt-1">
                        {errorCount} file{errorCount !== 1 ? 's' : ''} failed
                      </p>
                    )}
                  </div>
                )}

                {/* File progress list */}
                {fileEntries.length > 0 && (
                  <div className="mt-3">
                    <FileProgressList entries={fileEntries} />
                  </div>
                )}
              </div>
            </div>
            <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
              <button
                type="button"
                onClick={handleCancel}
                className={`inline-flex w-full justify-center rounded-md px-3 py-2 text-sm font-semibold text-white shadow-xs sm:ml-3 sm:w-auto ${
                  isUploading ? 'bg-red-600 hover:bg-red-500' : 'bg-gray-600 hover:bg-gray-500'
                }`}
              >
                {isUploading ? 'Cancel Upload' : 'Close'}
              </button>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}

/** Virtualized file progress list */
function FileProgressList({ entries }: { entries: FileEntry[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);

  const totalHeight = entries.length * ROW_HEIGHT;
  const containerHeight = Math.min(entries.length, VISIBLE_ROWS) * ROW_HEIGHT;

  const startIdx = Math.floor(scrollTop / ROW_HEIGHT);
  const endIdx = Math.min(entries.length, startIdx + VISIBLE_ROWS + 2);
  const visibleEntries = entries.slice(startIdx, endIdx);
  const offsetY = startIdx * ROW_HEIGHT;

  const onScroll = useCallback(() => {
    if (containerRef.current) {
      setScrollTop(containerRef.current.scrollTop);
    }
  }, []);

  return (
    <div
      ref={containerRef}
      onScroll={onScroll}
      className="rounded-xl bg-black/60 backdrop-blur-sm border border-white/10 overflow-y-auto"
      style={{ height: containerHeight + 2 }}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        <div style={{ position: 'absolute', top: offsetY, left: 0, right: 0 }}>
          {visibleEntries.map(entry => (
            <FileRow key={entry.id} entry={entry} />
          ))}
        </div>
      </div>
    </div>
  );
}

function FileRow({ entry }: { entry: FileEntry }) {
  return (
    <div className="flex items-center gap-2 px-3 text-xs font-mono" style={{ height: ROW_HEIGHT }}>
      <span className="flex-shrink-0 w-4 text-center">
        {entry.status === 'error' && <FaTimesCircle className="text-red-400 inline" />}
        {entry.status === 'uploading' && <FaSpinner className="text-blue-400 inline animate-spin" />}
        {entry.status === 'pending' && <span className="inline-block w-2 h-2 rounded-full bg-white/30" />}
      </span>
      <span className="truncate flex-1 opacity-80" title={entry.file.name}>
        {entry.file.name}
      </span>
      <span className="flex-shrink-0 w-16 text-right">
        {entry.status === 'uploading' && <span className="text-blue-300">{entry.progress}%</span>}
        {entry.status === 'error' && <span className="text-red-400">Failed</span>}
        {entry.status === 'pending' && <span className="text-white/30">Queued</span>}
      </span>
    </div>
  );
}
