'use client';
import React, { useRef, useEffect, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaCut, FaCodeBranch } from 'react-icons/fa';
import { apiClient } from '@/utils/api';

interface VideoTrimModalProps {
  videoUrl: string;
  isOpen: boolean;
  onClose: () => void;
  onTrim?: () => void;
  onSplit?: () => void;
}

const formatTime = (seconds: number): string => {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 10);
  if (h > 0) {
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
  }
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
};

const parseTimeInput = (value: string): number | null => {
  const trimmed = value.trim();
  const asFloat = parseFloat(trimmed);
  if (!isNaN(asFloat) && !trimmed.includes(':')) return asFloat;
  const parts = trimmed.split(':');
  if (parts.length === 2) {
    const mins = parseInt(parts[0], 10);
    const secs = parseFloat(parts[1]);
    if (!isNaN(mins) && !isNaN(secs)) return mins * 60 + secs;
  }
  if (parts.length === 3) {
    const hours = parseInt(parts[0], 10);
    const mins = parseInt(parts[1], 10);
    const secs = parseFloat(parts[2]);
    if (!isNaN(hours) && !isNaN(mins) && !isNaN(secs)) return hours * 3600 + mins * 60 + secs;
  }
  return null;
};

export default function VideoTrimModal({ videoUrl, isOpen, onClose, onTrim, onSplit }: VideoTrimModalProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [mounted, setMounted] = useState(false);
  const [duration, setDuration] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [startInput, setStartInput] = useState('00:00.0');
  const [endInput, setEndInput] = useState('00:00.0');
  const [activeTab, setActiveTab] = useState<'trim' | 'split'>('trim');
  const [splitSeconds, setSplitSeconds] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!isOpen) {
      setError(null);
      setIsProcessing(false);
      if (videoRef.current) {
        videoRef.current.pause();
      }
    }
  }, [isOpen]);

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    const dur = video.duration;
    setDuration(dur);
    setEndTime(dur);
    setEndInput(formatTime(dur));
    setStartTime(0);
    setStartInput('00:00.0');
  };

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.currentTime >= endTime || video.currentTime < startTime) {
      video.currentTime = startTime;
    }
  }, [startTime, endTime]);

  const handleStartSliderChange = (val: number) => {
    const clamped = Math.min(val, endTime - 0.1);
    setStartTime(clamped);
    setStartInput(formatTime(clamped));
    if (videoRef.current) {
      videoRef.current.currentTime = clamped;
    }
  };

  const handleEndSliderChange = (val: number) => {
    const clamped = Math.max(val, startTime + 0.1);
    setEndTime(clamped);
    setEndInput(formatTime(clamped));
    if (videoRef.current) {
      videoRef.current.currentTime = clamped;
    }
  };

  const handleStartInputBlur = () => {
    const parsed = parseTimeInput(startInput);
    if (parsed !== null && parsed >= 0 && parsed < endTime) {
      setStartTime(parsed);
      setStartInput(formatTime(parsed));
      if (videoRef.current) videoRef.current.currentTime = parsed;
    } else {
      setStartInput(formatTime(startTime));
    }
  };

  const handleEndInputBlur = () => {
    const parsed = parseTimeInput(endInput);
    if (parsed !== null && parsed > startTime && parsed <= duration) {
      setEndTime(parsed);
      setEndInput(formatTime(parsed));
      if (videoRef.current) videoRef.current.currentTime = parsed;
    } else {
      setEndInput(formatTime(endTime));
    }
  };

  const handleClose = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.pause();
    }
    onClose();
  }, [onClose]);

  const handleTrim = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      await apiClient.post('/api/video/trim', { videoPath: videoUrl, startTime, endTime });
      onTrim?.();
      handleClose();
    } catch (err: any) {
      setError(err?.response?.data?.error || 'Failed to trim video');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSplit = async () => {
    const seconds = parseInt(splitSeconds, 10);
    if (isNaN(seconds) || seconds < 1) {
      setError('Please enter a valid number of seconds (minimum 1).');
      return;
    }
    setIsProcessing(true);
    setError(null);
    try {
      await apiClient.post('/api/video/split', { videoPath: videoUrl, secondsPerSegment: seconds });
      onSplit?.();
      handleClose();
    } catch (err: any) {
      setError(err?.response?.data?.error || 'Failed to split video');
    } finally {
      setIsProcessing(false);
    }
  };

  if (!mounted) return null;

  const videoSrc = `/api/img/${encodeURIComponent(videoUrl)}`;

  return createPortal(
    <Dialog open={isOpen} onClose={handleClose} className="relative z-50">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div className="fixed inset-0 z-10 w-screen overflow-y-auto"
        onPointerDown={e => e.stopPropagation()}
        onPointerUp={e => e.stopPropagation()}
        onPointerLeave={e => e.stopPropagation()}
        onPointerCancel={e => e.stopPropagation()}
      >
        <div className="flex min-h-full items-center justify-center p-4 text-center">
          <DialogPanel
            transition
            className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 w-full max-w-3xl data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="bg-gray-800 px-6 pt-5 pb-4">
              <DialogTitle as="h3" className="text-base font-semibold text-gray-100 mb-4">
                Edit Video
              </DialogTitle>

              {/* Video Preview */}
              <div className="mb-4 bg-black rounded-lg overflow-hidden flex items-center justify-center">
                <video
                  ref={videoRef}
                  src={videoSrc}
                  className="w-full max-h-64 object-contain"
                  controls
                  onLoadedMetadata={handleLoadedMetadata}
                  onTimeUpdate={handleTimeUpdate}
                />
              </div>

              {/* Tabs */}
              <div className="flex border-b border-gray-700 mb-4">
                <button
                  className={`px-4 py-2 text-sm font-medium flex items-center gap-2 ${activeTab === 'trim' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-200'}`}
                  onClick={() => { setActiveTab('trim'); setError(null); }}
                >
                  <FaCut />
                  Trim
                </button>
                <button
                  className={`px-4 py-2 text-sm font-medium flex items-center gap-2 ${activeTab === 'split' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-400 hover:text-gray-200'}`}
                  onClick={() => { setActiveTab('split'); setError(null); }}
                >
                  <FaCodeBranch />
                  Split
                </button>
              </div>

              {/* Trim Tab */}
              {activeTab === 'trim' && (
                <div className="space-y-4">
                  <p className="text-sm text-gray-300">
                    Select the start and end time to trim the video down to. The video preview above will loop within the selected range.
                  </p>

                  {/* Start time */}
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Start Time</label>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min={0}
                        max={duration || 1}
                        step={0.1}
                        value={startTime}
                        onChange={e => handleStartSliderChange(parseFloat(e.target.value))}
                        className="flex-1 accent-blue-500"
                        aria-label="Start time"
                      />
                      <input
                        type="text"
                        value={startInput}
                        onChange={e => setStartInput(e.target.value)}
                        onBlur={handleStartInputBlur}
                        className="w-24 bg-gray-700 text-white text-sm rounded px-2 py-1 text-center"
                        aria-label="Start time value"
                      />
                    </div>
                  </div>

                  {/* End time */}
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">End Time</label>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min={0}
                        max={duration || 1}
                        step={0.1}
                        value={endTime}
                        onChange={e => handleEndSliderChange(parseFloat(e.target.value))}
                        className="flex-1 accent-blue-500"
                        aria-label="End time"
                      />
                      <input
                        type="text"
                        value={endInput}
                        onChange={e => setEndInput(e.target.value)}
                        onBlur={handleEndInputBlur}
                        className="w-24 bg-gray-700 text-white text-sm rounded px-2 py-1 text-center"
                        aria-label="End time value"
                      />
                    </div>
                  </div>

                  <p className="text-xs text-gray-500">
                    Trimmed duration: {formatTime(Math.max(0, endTime - startTime))} (from {formatTime(startTime)} to {formatTime(endTime)} of {formatTime(duration)})
                  </p>
                </div>
              )}

              {/* Split Tab */}
              {activeTab === 'split' && (
                <div className="space-y-4">
                  <p className="text-sm text-gray-300">
                    Split the video into segments of equal length. The original video will be replaced by the individual segments.
                  </p>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Seconds per segment</label>
                    <input
                      type="number"
                      min={1}
                      value={splitSeconds}
                      onChange={e => setSplitSeconds(e.target.value)}
                      placeholder="e.g. 30"
                      className="w-full bg-gray-700 text-white text-sm rounded px-3 py-2"
                      aria-label="Seconds per segment"
                    />
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-3 text-sm text-red-400">{error}</div>
              )}
            </div>

            <div className="bg-gray-700 px-6 py-3 flex justify-end gap-3">
              <button
                type="button"
                onClick={handleClose}
                className="rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-900"
              >
                Cancel
              </button>
              {activeTab === 'trim' && (
                <button
                  type="button"
                  onClick={handleTrim}
                  disabled={isProcessing || duration === 0}
                  className="rounded-md bg-blue-700 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed px-3 py-2 text-sm font-semibold text-white"
                >
                  {isProcessing ? 'Trimming…' : 'Trim Video'}
                </button>
              )}
              {activeTab === 'split' && (
                <button
                  type="button"
                  onClick={handleSplit}
                  disabled={isProcessing}
                  className="rounded-md bg-blue-700 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed px-3 py-2 text-sm font-semibold text-white"
                >
                  {isProcessing ? 'Splitting…' : 'Split Video'}
                </button>
              )}
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
}
