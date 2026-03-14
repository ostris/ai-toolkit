'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { FaChevronLeft, FaChevronRight, FaArrowsAlt } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import { isVideo } from '@/utils/basic';
import GalleryCopyModal from './GalleryCopyModal';
import MoveImageModal from './MoveImageModal';

interface ImagePair {
  left: string;
  right: string;
  filename: string;
}

interface CompareViewProps {
  pairs: ImagePair[];
  leftLabel: string;
  rightLabel: string;
  mode: 'dataset' | 'gallery';
  /** For dataset mode, the left dataset name (used by MoveImageModal) */
  leftDataset?: string;
  /** For dataset mode, the right dataset name (used by MoveImageModal) */
  rightDataset?: string;
}

function CompareImagePanel({
  imgPath,
  mode,
  dataset,
}: {
  imgPath: string;
  mode: 'dataset' | 'gallery';
  dataset?: string;
}) {
  const [caption, setCaption] = useState('');
  const [savedCaption, setSavedCaption] = useState('');
  const [isCaptionLoaded, setIsCaptionLoaded] = useState(false);
  const [isCopyOpen, setIsCopyOpen] = useState(false);
  const isGettingCaption = useRef(false);

  const fetchCaption = useCallback(() => {
    if (isGettingCaption.current) return;
    isGettingCaption.current = true;
    const endpoint = mode === 'dataset' ? '/api/caption/get' : '/api/gallery/caption';
    apiClient
      .post(endpoint, { imgPath })
      .then(res => {
        const data = res.data ? `${res.data}` : '';
        setCaption(data);
        setSavedCaption(data);
        setIsCaptionLoaded(true);
      })
      .catch(() => {
        setIsCaptionLoaded(true);
      })
      .finally(() => {
        isGettingCaption.current = false;
      });
  }, [imgPath, mode]);

  useEffect(() => {
    setIsCaptionLoaded(false);
    setCaption('');
    setSavedCaption('');
    isGettingCaption.current = false;
    fetchCaption();
  }, [imgPath, fetchCaption]);

  const saveCaption = () => {
    const trimmed = caption.trim();
    if (trimmed === savedCaption) return;
    const endpoint = mode === 'dataset' ? '/api/img/caption' : '/api/gallery/captionSave';
    apiClient
      .post(endpoint, { imgPath, caption: trimmed })
      .then(() => setSavedCaption(trimmed))
      .catch(err => console.error('Error saving caption:', err));
  };

  const imgApiBase = mode === 'dataset' ? '/api/img' : '/api/gallery/img';
  const isCaptionCurrent = caption.trim() === savedCaption;
  const isItVideo = isVideo(imgPath);

  return (
    <div className="flex flex-col flex-1 min-w-0">
      {/* Image */}
      <div className="relative bg-gray-900 rounded-t-lg flex items-center justify-center" style={{ minHeight: '300px', maxHeight: '60vh' }}>
        {isItVideo ? (
          <video
            key={imgPath}
            src={`${imgApiBase}/${encodeURIComponent(imgPath)}`}
            className="max-w-full max-h-[60vh] object-contain"
            controls
            loop
          />
        ) : (
          <img
            key={imgPath}
            src={`${imgApiBase}/${encodeURIComponent(imgPath)}`}
            alt={imgPath}
            className="max-w-full max-h-[60vh] object-contain"
          />
        )}
        {/* Copy to dataset button */}
        <button
          className="absolute top-2 right-2 bg-gray-800 hover:bg-gray-700 rounded-full p-2 transition-colors z-10"
          onClick={() => setIsCopyOpen(true)}
          aria-label="Copy to dataset"
          title="Copy to dataset"
        >
          <FaArrowsAlt className="text-gray-200" />
        </button>
      </div>

      {/* Caption */}
      <div className="relative w-full" style={{ height: '75px' }}>
        <div
          className={`absolute inset-x-0 top-0 p-2 bg-gray-800 text-white text-sm rounded-b-lg h-[75px] hover:h-[150px] transition-[height] duration-300 ease-in-out z-20 overflow-hidden border-2 ${
            isCaptionCurrent ? 'border-transparent' : 'border-blue-500'
          }`}
        >
          {isCaptionLoaded ? (
            <form
              className="h-full"
              onSubmit={e => { e.preventDefault(); saveCaption(); }}
              onBlur={saveCaption}
            >
              <textarea
                className="w-full h-full bg-transparent resize-none outline-none focus:ring-0 focus:outline-none"
                value={caption}
                onChange={e => setCaption(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    saveCaption();
                  }
                  // Prevent arrow key navigation when editing caption
                  e.stopPropagation();
                }}
              />
            </form>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-gray-400">
              Loading caption...
            </div>
          )}
        </div>
      </div>

      {/* Copy modal */}
      {mode === 'gallery' ? (
        <GalleryCopyModal
          isOpen={isCopyOpen}
          onClose={() => setIsCopyOpen(false)}
          imageUrl={imgPath}
          onComplete={() => setIsCopyOpen(false)}
        />
      ) : (
        <MoveImageModal
          isOpen={isCopyOpen}
          onClose={() => setIsCopyOpen(false)}
          imageUrl={imgPath}
          currentDataset={dataset || ''}
          onComplete={() => setIsCopyOpen(false)}
        />
      )}
    </div>
  );
}

export default function CompareView({
  pairs,
  leftLabel,
  rightLabel,
  mode,
  leftDataset,
  rightDataset,
}: CompareViewProps) {
  const [currentIndex, setCurrentIndex] = useState(0);

  const goToPrev = useCallback(() => {
    setCurrentIndex(prev => Math.max(0, prev - 1));
  }, []);

  const goToNext = useCallback(() => {
    setCurrentIndex(prev => Math.min(pairs.length - 1, prev + 1));
  }, [pairs.length]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle arrow keys if user is typing in a textarea/input
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'TEXTAREA' || tag === 'INPUT') return;

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        goToPrev();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        goToNext();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goToPrev, goToNext]);

  if (pairs.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        No images to compare.
      </div>
    );
  }

  const pair = pairs[currentIndex];

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Labels */}
      <div className="flex gap-4">
        <div className="flex-1 text-center">
          <span className="text-sm font-medium text-gray-300 bg-gray-800 px-3 py-1 rounded-full">
            {leftLabel}
          </span>
        </div>
        <div className="flex-1 text-center">
          <span className="text-sm font-medium text-gray-300 bg-gray-800 px-3 py-1 rounded-full">
            {rightLabel}
          </span>
        </div>
      </div>

      {/* Side-by-side images */}
      <div className="flex gap-4">
        <CompareImagePanel
          imgPath={pair.left}
          mode={mode}
          dataset={leftDataset}
        />
        <CompareImagePanel
          imgPath={pair.right}
          mode={mode}
          dataset={rightDataset}
        />
      </div>

      {/* Filename display */}
      <div className="text-center text-xs text-gray-500">
        {pair.filename}
      </div>

      {/* Navigation controls */}
      <div className="flex items-center justify-center gap-4">
        <button
          onClick={goToPrev}
          disabled={currentIndex <= 0}
          className="flex items-center gap-1 px-4 py-2 bg-gray-700 text-gray-200 rounded-md hover:bg-gray-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          <FaChevronLeft className="w-3 h-3" />
          Previous
        </button>

        {/* Progress indicator */}
        <div className="flex items-center gap-2">
          <span className="text-gray-200 font-medium text-lg">
            {currentIndex + 1}
          </span>
          <span className="text-gray-500">/</span>
          <span className="text-gray-400 text-lg">
            {pairs.length}
          </span>
        </div>

        <button
          onClick={goToNext}
          disabled={currentIndex >= pairs.length - 1}
          className="flex items-center gap-1 px-4 py-2 bg-gray-700 text-gray-200 rounded-md hover:bg-gray-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          Next
          <FaChevronRight className="w-3 h-3" />
        </button>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-md mx-auto">
        <div className="w-full bg-gray-700 rounded-full h-1.5">
          <div
            className="bg-blue-500 h-1.5 rounded-full transition-all duration-200"
            style={{ width: `${((currentIndex + 1) / pairs.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Keyboard hint */}
      <div className="text-center text-xs text-gray-600">
        Use arrow keys to navigate
      </div>
    </div>
  );
}
