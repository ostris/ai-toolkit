'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { FaUpload, FaTimesCircle, FaSpinner } from 'react-icons/fa';
import { apiClient } from '@/utils/api';

type AcceptMap = {
  [mime: string]: string[];
};

interface FullscreenDropOverlayProps {
  datasetName: string;
  onComplete?: () => void;
  accept?: AcceptMap;
  multiple?: boolean;
}

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

export default function FullscreenDropOverlay({
  datasetName,
  onComplete,
  accept,
  multiple = true,
}: FullscreenDropOverlayProps) {
  const [visible, setVisible] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [fileEntries, setFileEntries] = useState<FileEntry[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [doneCount, setDoneCount] = useState(0);
  const [errorCount, setErrorCount] = useState(0);
  const dragDepthRef = useRef(0);
  const abortRef = useRef(false);

  const isFileDrag = (e: DragEvent) => {
    const types = e?.dataTransfer?.types;
    return !!types && Array.from(types).includes('Files');
  };

  useEffect(() => {
    const onDragEnter = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      dragDepthRef.current += 1;
      setVisible(true);
      e.preventDefault();
    };
    const onDragOver = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      e.preventDefault();
      if (!visible) setVisible(true);
    };
    const onDragLeave = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
      if (dragDepthRef.current === 0 && !isUploading) {
        setVisible(false);
      }
    };
    const onWindowDrop = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      e.preventDefault();
      dragDepthRef.current = 0;
    };

    window.addEventListener('dragenter', onDragEnter);
    window.addEventListener('dragover', onDragOver);
    window.addEventListener('dragleave', onDragLeave);
    window.addEventListener('drop', onWindowDrop);

    return () => {
      window.removeEventListener('dragenter', onDragEnter);
      window.removeEventListener('dragover', onDragOver);
      window.removeEventListener('dragleave', onDragLeave);
      window.removeEventListener('drop', onWindowDrop);
    };
  }, [visible, isUploading]);

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
        // Remove from list on success
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
        onComplete?.();
        setFileEntries([]);
        setVisible(false);
        setTotalCount(0);
        setDoneCount(0);
        setErrorCount(0);
      }
    },
    [uploadSingleFile, onComplete],
  );

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) {
        setVisible(false);
        return;
      }

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

  const handleClose = useCallback(() => {
    abortRef.current = true;
    setIsUploading(false);
    setFileEntries([]);
    setVisible(false);
    setTotalCount(0);
    setDoneCount(0);
    setErrorCount(0);
  }, []);

  const dropAccept = useMemo<AcceptMap>(
    () =>
      accept || {
        'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'],
        'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'],
        'audio/*': ['.mp3', '.wav'],
        'text/*': ['.txt'],
      },
    [accept],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: dropAccept,
    multiple,
    noClick: true,
    noKeyboard: true,
    preventDropOnDocument: true,
  });

  const overallPercent = totalCount > 0 ? Math.round(((doneCount + errorCount) / totalCount) * 100) : 0;

  return (
    <div
      className={`fixed inset-0 z-[9999] transition-opacity duration-200 ${
        visible || isUploading ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
      }`}
      aria-hidden={!visible && !isUploading}
      {...getRootProps()}
    >
      <input {...getInputProps()} />
      <div className={`absolute inset-0 ${isUploading ? 'bg-gray-900/70' : 'bg-gray-900/40'}`} />

      <div className="absolute inset-0 flex items-center justify-center p-6">
        <div className="w-full max-w-2xl flex flex-col gap-3">
          {/* Drop target / status box */}
          <div
            className={`rounded-2xl border-2 border-dashed px-8 py-10 text-center shadow-2xl backdrop-blur-sm
            ${isDragActive ? 'border-blue-400 bg-white/10' : 'border-white/30 bg-white/5'}`}
          >
            <div className="flex flex-col items-center gap-4">
              <FaUpload className="size-10 opacity-80" />
              {!isUploading ? (
                <>
                  <p className="text-lg font-semibold">Drop files to upload</p>
                  <p className="text-sm opacity-80">
                    Destination:&nbsp;<span className="font-mono">{datasetName || 'unknown'}</span>
                  </p>
                  <p className="text-xs opacity-70 mt-1">Images, videos, or .txt supported</p>
                </>
              ) : (
                <>
                  <p className="text-lg font-semibold">
                    Uploading… {doneCount + errorCount} / {totalCount}
                  </p>
                  <div className="w-full h-2.5 bg-white/20 rounded-full overflow-hidden">
                    <div
                      className="h-2.5 bg-blue-500 rounded-full transition-[width] duration-150 ease-linear"
                      style={{ width: `${overallPercent}%` }}
                    />
                  </div>
                  {errorCount > 0 && (
                    <p className="text-xs text-red-400">
                      {errorCount} file{errorCount !== 1 ? 's' : ''} failed
                    </p>
                  )}
                  <button
                    type="button"
                    onClick={e => {
                      e.stopPropagation();
                      handleClose();
                    }}
                    className="mt-2 px-4 py-1.5 text-sm rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                  >
                    Cancel
                  </button>
                </>
              )}
            </div>
          </div>

          {/* File progress list — only shows pending, uploading, and errored files */}
          {fileEntries.length > 0 && <FileProgressList entries={fileEntries} />}
        </div>
      </div>
    </div>
  );
}

/** Virtualized file progress list for handling thousands of files */
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
      onClick={e => e.stopPropagation()}
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
