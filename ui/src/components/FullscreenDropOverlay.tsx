'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { FaUpload } from 'react-icons/fa';
import { apiClient } from '@/utils/api';

type AcceptMap = {
  [mime: string]: string[];
};

interface FullscreenDropOverlayProps {
  datasetName: string; // where to upload
  onComplete?: () => void; // called after successful upload
  accept?: AcceptMap; // optional override
  multiple?: boolean; // default true
}

export default function FullscreenDropOverlay({
  datasetName,
  onComplete,
  accept,
  multiple = true,
}: FullscreenDropOverlayProps) {
  const [visible, setVisible] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const dragDepthRef = useRef(0); // drag-enter/leave tracking

  // Only show the overlay for real file drags (not text, images from page, etc)
  const isFileDrag = (e: DragEvent) => {
    const types = e?.dataTransfer?.types;
    return !!types && Array.from(types).includes('Files');
  };

  // Window-level drag listeners to toggle visibility
  useEffect(() => {
    const onDragEnter = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      dragDepthRef.current += 1;
      setVisible(true);
      e.preventDefault();
    };
    const onDragOver = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      // Must preventDefault to allow dropping in the browser
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
    const onDrop = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      // Prevent browser from opening the file
      e.preventDefault();
      dragDepthRef.current = 0;
      // We do NOT hide here; the dropzone onDrop will handle workflow visibility.
    };

    window.addEventListener('dragenter', onDragEnter);
    window.addEventListener('dragover', onDragOver);
    window.addEventListener('dragleave', onDragLeave);
    window.addEventListener('drop', onDrop);

    return () => {
      window.removeEventListener('dragenter', onDragEnter);
      window.removeEventListener('dragover', onDragOver);
      window.removeEventListener('dragleave', onDragLeave);
      window.removeEventListener('drop', onDrop);
    };
  }, [visible, isUploading]);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) {
        // no accepted files; hide overlay cleanly
        setVisible(false);
        return;
      }

      setIsUploading(true);
      setUploadProgress(0);

      const formData = new FormData();
      acceptedFiles.forEach(file => formData.append('files', file));
      formData.append('datasetName', datasetName || '');

      try {
        await apiClient.post(`/api/datasets/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: pe => {
            const percent = Math.round(((pe.loaded || 0) * 100) / (pe.total || pe.loaded || 1));
            setUploadProgress(percent);
          },
          timeout: 0,
        });
        onComplete?.();
      } catch (err) {
        console.error('Upload failed:', err);
      } finally {
        setIsUploading(false);
        setUploadProgress(0);
        setVisible(false);
      }
    },
    [datasetName, onComplete],
  );

  const dropAccept = useMemo<AcceptMap>(
    () =>
      accept || {
        'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'],
        'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'],
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
    // Prevent "folder opens" by browser if someone drags outside the overlay mid-drop:
    preventDropOnDocument: true,
  });

  return (
    <div
      // When hidden: opacity-0 + pointer-events-none so the page is fully interactive
      // When visible or uploading: fade in and capture the drop
      className={`fixed inset-0 z-[9999] transition-opacity duration-200 ${
        visible || isUploading ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
      }`}
      aria-hidden={!visible && !isUploading}
      {...getRootProps()}
    >
      {/* Fullscreen capture layer */}
      <input {...getInputProps()} />

      {/* Backdrop: keep it subtle so context remains visible */}
      <div className={`absolute inset-0 ${isUploading ? 'bg-gray-900/70' : 'bg-gray-900/40'}`} />

      {/* Center drop target UI */}
      <div className="absolute inset-0 flex items-center justify-center p-6">
        <div
          className={`w-full max-w-2xl rounded-2xl border-2 border-dashed px-8 py-10 text-center shadow-2xl backdrop-blur-sm
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
                <p className="text-lg font-semibold">Uploading… {uploadProgress}%</p>
                <div className="w-full h-2.5 bg-white/20 rounded-full overflow-hidden">
                  <div
                    className="h-2.5 bg-blue-500 rounded-full transition-[width] duration-150 ease-linear"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
