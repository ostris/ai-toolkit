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
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setSecondsPerSegment('');
      setError(null);
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
    try {
      const results = await Promise.allSettled(
        videoPaths.map(videoPath =>
          apiClient.post('/api/video/split', { videoPath, secondsPerSegment: seconds }),
        ),
      );
      const failures = results
        .map((r, i) => ({ result: r, path: videoPaths[i] }))
        .filter(({ result }) => result.status === 'rejected');
      if (failures.length > 0) {
        const names = failures.map(({ path }) => path.split(/[\\/]/).pop()).join(', ');
        setError(`Failed to split: ${names}`);
      } else {
        onComplete(videoPaths);
        onClose();
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
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
            className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="Seconds per segment"
            autoFocus
          />
        </div>
        {error && <p className="text-red-400 text-sm">{error}</p>}
        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none"
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
