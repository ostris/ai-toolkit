'use client';

import React, { useCallback, useMemo, useRef, useState } from 'react';
import classNames from 'classnames';
import { useDropzone } from 'react-dropzone';
import { FaUpload, FaImage, FaTimes } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import type { AxiosProgressEvent } from 'axios';

interface Props {
  src: string | null | undefined;
  className?: string;
  instruction?: string;
  onNewImageSelected: (imagePath: string | null) => void;
}

export default function SampleControlImage({
  src,
  className,
  instruction = 'Add Control Image',
  onNewImageSelected,
}: Props) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [localPreview, setLocalPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const backgroundUrl = useMemo(() => {
    if (localPreview) return localPreview;
    if (src) return `/api/img/${encodeURIComponent(src)}`;
    return null;
  }, [src, localPreview]);

  const handleUpload = useCallback(
    async (file: File) => {
      if (!file) return;
      setIsUploading(true);
      setUploadProgress(0);

      const objectUrl = URL.createObjectURL(file);
      setLocalPreview(objectUrl);

      const formData = new FormData();
      formData.append('files', file);

      try {
        const resp = await apiClient.post(`/api/img/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (evt: AxiosProgressEvent) => {
            const total = evt.total ?? 100;
            const loaded = evt.loaded ?? 0;
            setUploadProgress(Math.round((loaded * 100) / total));
          },
          timeout: 0,
        });

        const uploaded = resp?.data?.files?.[0] ?? null;
        onNewImageSelected(uploaded);
      } catch (err) {
        console.error('Upload failed:', err);
        setLocalPreview(null);
      } finally {
        setIsUploading(false);
        setUploadProgress(0);
        URL.revokeObjectURL(objectUrl);
        if (fileInputRef.current) fileInputRef.current.value = '';
      }
    },
    [onNewImageSelected],
  );

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      handleUpload(acceptedFiles[0]);
    },
    [handleUpload],
  );

  const clearImage = useCallback(
    (e?: React.MouseEvent) => {
      console.log('clearImage');
      if (e) {
        e.stopPropagation();
        e.preventDefault();
      }
      setLocalPreview(null);
      onNewImageSelected(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    },
    [onNewImageSelected],
  );

  // Drag & drop only; click handled via our own hidden input
  const { getRootProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'] },
    multiple: false,
    noClick: true,
    noKeyboard: true,
  });

  const rootProps = getRootProps();

  return (
    <div
      {...rootProps}
      className={classNames(
        'group relative flex items-center justify-center rounded-xl cursor-pointer ring-1 ring-inset',
        'transition-all duration-200 select-none overflow-hidden text-center',
        'h-20 w-20',
        backgroundUrl ? 'bg-gray-800 ring-gray-700' : 'bg-gradient-to-b from-gray-800 to-gray-900 ring-gray-700',
        isDragActive ? 'outline outline-2 outline-blue-500' : 'hover:ring-gray-600',
        className,
      )}
      style={
        backgroundUrl
          ? {
              backgroundImage: `url("${backgroundUrl}")`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
            }
          : undefined
      }
      onClick={() => !isUploading && fileInputRef.current?.click()}
    >
      {/* Hidden input for click-to-open */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={e => {
          const file = e.currentTarget.files?.[0];
          if (file) handleUpload(file);
        }}
      />

      {/* Empty state — centered */}
      {!backgroundUrl && (
        <div className="flex flex-col items-center justify-center text-gray-300 text-center">
          <FaImage className="opacity-80" />
          <div className="mt-1 text-[10px] font-semibold tracking-wide opacity-80">{instruction}</div>
          <div className="mt-0.5 text-[9px] opacity-60">Click or drop</div>
        </div>
      )}

      {/* Existing image overlays */}
      {backgroundUrl && !isUploading && (
        <>
          <div
            className={classNames(
              'pointer-events-none absolute inset-0 flex items-center justify-center',
              'bg-black/0 group-hover:bg-black/20',
              isDragActive && 'bg-black/35',
              'transition-colors',
            )}
          >
            <div
              className={classNames(
                'inline-flex items-center gap-1 rounded-md px-2 py-1',
                'text-[10px] font-semibold',
                'bg-black/45 text-white/90 backdrop-blur-sm',
                'opacity-0 group-hover:opacity-100 transition-opacity',
              )}
            >
              <FaUpload className="text-[10px]" />
              <span>Replace</span>
            </div>
          </div>

          {/* Clear (X) button */}
          <button
            type="button"
            onClick={clearImage}
            title="Clear image"
            aria-label="Clear image"
            className={classNames(
              'absolute right-1.5 top-1.5 z-10 inline-flex items-center justify-center',
              'h-5 w-5 rounded-md bg-black/55 text-white/90',
              'opacity-0 group-hover:opacity-100 transition-opacity',
              'hover:bg-black/70',
            )}
          >
            <FaTimes className="text-[10px]" />
          </button>
        </>
      )}

      {/* Uploading overlay */}
      {isUploading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-[1px] text-center">
          <div className="w-4/5 max-w-40">
            <div className="h-1.5 w-full rounded-full bg-white/15">
              <div
                className="h-1.5 rounded-full bg-white/80 transition-[width]"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <div className="mt-1 text-[10px] font-medium text-white/90">Uploading… {uploadProgress}%</div>
          </div>
        </div>
      )}
    </div>
  );
}
