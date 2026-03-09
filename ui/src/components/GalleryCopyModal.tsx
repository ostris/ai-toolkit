import React, { useState, useEffect } from 'react';
import { Modal } from './Modal';
import { apiClient } from '@/utils/api';

interface GalleryCopyModalProps {
  isOpen: boolean;
  onClose: () => void;
  imageUrl: string;
  onComplete: () => void;
}

const GalleryCopyModal: React.FC<GalleryCopyModalProps> = ({ isOpen, onClose, imageUrl, onComplete }) => {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [targetDataset, setTargetDataset] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setError(null);
      setSuccess(false);
      apiClient
        .get('/api/datasets/list')
        .then(res => res.data)
        .then((data: string[]) => {
          setDatasets(data);
          setTargetDataset(data[0] || '');
        })
        .catch(() => {
          setDatasets([]);
          setError('Failed to load datasets');
        });
    }
  }, [isOpen]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!targetDataset) return;
    setIsLoading(true);
    setError(null);
    try {
      await apiClient.post('/api/gallery/copyToDataset', { imgPath: imageUrl, targetDataset });
      setSuccess(true);
      onComplete();
      onClose();
    } catch (err: any) {
      setError(err?.response?.data?.error || 'Failed to copy file');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Copy to Dataset" size="sm">
      <form onSubmit={handleSubmit} className="space-y-4 text-gray-200">
        {datasets.length === 0 ? (
          <>
            <p className="text-gray-400 text-sm">{error || 'No datasets available. Create a dataset first.'}</p>
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
                {isLoading ? 'Copying…' : 'Copy'}
              </button>
            </div>
          </>
        )}
      </form>
    </Modal>
  );
};

export default GalleryCopyModal;
