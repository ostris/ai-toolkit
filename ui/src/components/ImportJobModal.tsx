'use client';

import { useState, useCallback, useRef } from 'react';
import axios from 'axios';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaUpload, FaFileArchive, FaSpinner } from 'react-icons/fa';
import { useDropzone } from 'react-dropzone';
import { apiClient } from '@/utils/api';

interface ImportJobModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

type Phase = 'idle' | 'uploading' | 'extracting';

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export default function ImportJobModal({ isOpen, onClose, onSuccess }: ImportJobModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState('');
  const [phase, setPhase] = useState<Phase>('idle');
  const [uploadedBytes, setUploadedBytes] = useState(0);
  const [totalBytes, setTotalBytes] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  const busy = phase !== 'idle';

  const handleClose = () => {
    if (phase === 'extracting') return; // can't cancel server-side extraction
    if (phase === 'uploading') abortRef.current?.abort();
    setSelectedFile(null);
    setError('');
    setPhase('idle');
    onClose();
  };

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length > 0) {
      setSelectedFile(accepted[0]);
      setError('');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open: openFilePicker } = useDropzone({
    onDrop,
    accept: { 'application/zip': ['.zip'] },
    multiple: false,
    disabled: busy,
    noClick: true,
    noKeyboard: true,
  });

  const handleImport = async () => {
    if (!selectedFile) { setError('Please select a ZIP file to import.'); return; }

    const controller = new AbortController();
    abortRef.current = controller;

    setPhase('uploading');
    setUploadedBytes(0);
    setTotalBytes(selectedFile.size);
    setError('');

    try {
      const formData = new FormData();
      formData.append('zip', selectedFile);

      await apiClient.post('/api/jobs/import', formData, {
        signal: controller.signal,
        timeout: 0,
        onUploadProgress: (pe) => {
          const loaded = pe.loaded ?? 0;
          const total = pe.total ?? selectedFile.size;
          setUploadedBytes(loaded);
          setTotalBytes(total);
          if (loaded >= total) setPhase('extracting');
        },
      });

      onSuccess();
      setSelectedFile(null);
      setPhase('idle');
      onClose();
    } catch (e: any) {
      if (axios.isCancel(e)) return;
      setPhase('idle');
      setError(e.response?.data?.error ?? e.message ?? 'Import failed');
    }
  };

  const uploadPercent = totalBytes > 0 ? Math.round((uploadedBytes / totalBytes) * 100) : 0;

  return (
    <Dialog open={isOpen} onClose={busy ? () => {} : handleClose} className="relative z-10">
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
                  Import Job
                </DialogTitle>

                {/* Drop zone */}
                <div {...getRootProps()} className="w-full">
                  <input {...getInputProps()} />
                  <div
                    onClick={() => { if (!busy) openFilePicker(); }}
                    className={`h-40 w-full flex flex-col items-center justify-center border-2 border-dashed rounded-lg cursor-pointer transition-colors
                      ${isDragActive ? 'border-blue-400 bg-blue-500/10' : selectedFile ? 'border-green-500 bg-green-500/10' : 'border-gray-600 hover:border-gray-400'}
                      ${busy ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {selectedFile ? (
                      <>
                        <FaFileArchive className="size-8 mb-3 text-green-400" />
                        <p className="text-sm text-gray-200 text-center font-medium">{selectedFile.name}</p>
                        <p className="text-xs text-gray-400 mt-1">{formatBytes(selectedFile.size)}</p>
                        {!busy && <p className="text-xs text-gray-500 mt-1">Click or drag to replace</p>}
                      </>
                    ) : (
                      <>
                        <FaUpload className="size-8 mb-3 text-gray-400" />
                        <p className="text-sm text-gray-200 text-center">
                          {isDragActive ? 'Drop file here…' : 'Drag & drop .zip file here or click to select'}
                        </p>
                        <p className="text-xs text-gray-400 mt-1">Only .zip files exported from AI Toolkit</p>
                      </>
                    )}
                  </div>
                </div>

                {/* Upload progress */}
                {phase === 'uploading' && (
                  <div className="mt-4">
                    <div className="flex justify-between text-sm mb-1.5">
                      <span className="text-gray-300">Uploading…</span>
                      <span className="text-gray-400 text-xs tabular-nums">
                        {formatBytes(uploadedBytes)} / {formatBytes(totalBytes)}
                      </span>
                    </div>
                    <div className="w-full h-2.5 bg-white/20 rounded-full overflow-hidden">
                      <div
                        className="h-2.5 bg-blue-500 rounded-full transition-[width] duration-150 ease-linear"
                        style={{ width: `${uploadPercent}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Extraction progress (indeterminate) */}
                {phase === 'extracting' && (
                  <div className="mt-4">
                    <div className="flex items-center gap-2 text-sm text-blue-400 mb-1.5">
                      <FaSpinner className="animate-spin flex-shrink-0" />
                      <span>Extracting… This may take a few minutes for large files.</span>
                    </div>
                    <div className="w-full h-2.5 bg-white/20 rounded-full overflow-hidden">
                      <div className="h-2.5 bg-blue-500/70 rounded-full animate-pulse w-full" />
                    </div>
                  </div>
                )}

                {/* Error */}
                {error && (
                  <p className="mt-3 text-sm text-red-400 bg-red-950/40 border border-red-800 rounded px-3 py-2 text-left">
                    {error}
                  </p>
                )}
              </div>
            </div>

            <div className="bg-gray-700 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
              <button
                type="button"
                onClick={handleImport}
                disabled={busy || !selectedFile}
                className="inline-flex w-full justify-center rounded-md bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed px-3 py-2 text-sm font-semibold text-white shadow-xs sm:ml-3 sm:w-auto"
              >
                {busy ? 'Importing…' : 'Import Job'}
              </button>
              <button
                type="button"
                onClick={handleClose}
                disabled={phase === 'extracting'}
                className="mt-3 inline-flex w-full justify-center rounded-md bg-gray-600 hover:bg-gray-500 disabled:opacity-50 disabled:cursor-not-allowed px-3 py-2 text-sm font-semibold text-gray-200 shadow-xs sm:mt-0 sm:w-auto"
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
