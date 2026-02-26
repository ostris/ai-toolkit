import React, { useState, useEffect } from 'react';
import { Modal } from './Modal';
import { apiClient } from '@/utils/api';

interface MoveImageModalProps {
  isOpen: boolean;
  onClose: () => void;
  imageUrl: string;
  currentDataset: string;
  onComplete: (operation: 'move' | 'copy') => void;
}

const MoveImageModal: React.FC<MoveImageModalProps> = ({ isOpen, onClose, imageUrl, currentDataset, onComplete }) => {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [targetDataset, setTargetDataset] = useState<string>('');
  const [operation, setOperation] = useState<'move' | 'copy'>('move');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      apiClient
        .get('/api/datasets/list')
        .then(res => res.data)
        .then((data: string[]) => {
          const filtered = data.filter(d => d !== currentDataset);
          setDatasets(filtered);
          setTargetDataset(filtered[0] || '');
        })
        .catch(() => {
          setDatasets([]);
          setError('Failed to load datasets');
        });
      setError(null);
    }
  }, [isOpen, currentDataset]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!targetDataset) return;
    setIsLoading(true);
    setError(null);
    try {
      await apiClient.post('/api/img/move', { imgPath: imageUrl, targetDataset, operation });
      onComplete(operation);
      onClose();
    } catch (err: any) {
      setError(err?.response?.data?.error || 'Failed to move/copy file');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Move / Copy to Dataset" size="sm">
      <form onSubmit={handleSubmit} className="space-y-4 text-gray-200">
        {datasets.length === 0 ? (
          <>
            <p className="text-gray-400 text-sm">{error || 'No other datasets available.'}</p>
            <div className="flex justify-end pt-2">
              <button
                type="button"
                className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none"
                onClick={onClose}
              >
                Close
              </button>
            </div>
          </>
        ) : (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Target Dataset</label>
              <select
                className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={targetDataset}
                onChange={e => setTargetDataset(e.target.value)}
              >
                {datasets.map(d => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Operation</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="operation"
                    value="move"
                    checked={operation === 'move'}
                    onChange={() => setOperation('move')}
                    className="accent-blue-500"
                  />
                  <span>Move</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="operation"
                    value="copy"
                    checked={operation === 'copy'}
                    onChange={() => setOperation('copy')}
                    className="accent-blue-500"
                  />
                  <span>Copy (duplicate)</span>
                </label>
              </div>
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
                disabled={isLoading || !targetDataset}
              >
                {isLoading ? 'Working…' : operation === 'move' ? 'Move' : 'Copy'}
              </button>
            </div>
          </>
        )}
      </form>
    </Modal>
  );
};

export default MoveImageModal;
