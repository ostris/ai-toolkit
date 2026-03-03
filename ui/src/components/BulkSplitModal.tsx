import React, { useState, useEffect } from 'react';
import { Modal } from './Modal';
import { apiClient } from '@/utils/api';

interface BulkSplitModalProps {
  isOpen: boolean;
  onClose: () => void;
  videoPaths: string[];
  onComplete: (splitPaths: string[]) => void;
}

const BulkSplitModal: React.FC<BulkSplitModalProps> = ({ isOpen, onClose, videoPaths, onComplete }) => {
  const [secondsPerSegment, setSecondsPerSegment] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState<{ completed: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setSecondsPerSegment('');
      setError(null);
      setProgress(null);
    }
  }, [isOpen]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const seconds = parseInt(secondsPerSegment, 10);
    if (isNaN(seconds) || seconds < 1) {
      setError('Please enter a valid number of seconds (minimum 1).');
      return;
    }
    setIsLoading(true);
    setError(null);
    setProgress({ completed: 0, total: videoPaths.length });

    const failed: string[] = [];
    for (const videoPath of videoPaths) {
      try {
        await apiClient.post('/api/video/split', { videoPath, secondsPerSegment: seconds });
      } catch (err) {
        console.error('Failed to split video:', videoPath, err);
        failed.push(videoPath.split(/[\\/]/).pop() ?? videoPath);
      }
      setProgress(prev => prev ? { ...prev, completed: prev.completed + 1 } : null);
    }

    setIsLoading(false);
    if (failed.length > 0) {
      setError(`Failed to split: ${failed.join(', ')}`);
      setProgress(null);
    } else {
      onComplete(videoPaths);
      onClose();
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={isLoading ? () => {} : onClose}
      title={`Split ${videoPaths.length} Video${videoPaths.length !== 1 ? 's' : ''}`}
      size="sm"
    >
      <form onSubmit={handleSubmit} className="space-y-4 text-gray-200">
        <p className="text-sm text-gray-400">
          Split the selected {videoPaths.length === 1 ? 'video' : `${videoPaths.length} videos`} into segments of equal
          length. Each original video will be replaced by its individual segments.
        </p>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Seconds per segment</label>
          <input
            type="number"
            min={1}
            value={secondsPerSegment}
            onChange={e => setSecondsPerSegment(e.target.value)}
            placeholder="e.g. 30"
            className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            aria-label="Seconds per segment"
            autoFocus
            disabled={isLoading}
          />
        </div>
        {progress && (
          <div>
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>Splitting videos…</span>
              <span>{progress.completed} / {progress.total}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: progress.total > 0 ? `${(progress.completed / progress.total) * 100}%` : '0%' }}
              />
            </div>
          </div>
        )}
        {error && <p className="text-red-400 text-sm">{error}</p>}
        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none disabled:opacity-50"
            onClick={onClose}
            disabled={isLoading}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none disabled:opacity-50"
            disabled={isLoading || !secondsPerSegment}
          >
            {isLoading ? 'Splitting…' : 'Split Videos'}
          </button>
        </div>
      </form>
    </Modal>
  );
};

export default BulkSplitModal;
