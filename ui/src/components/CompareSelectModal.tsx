'use client';

import React, { useState, useEffect } from 'react';
import { Modal } from './Modal';
import { apiClient } from '@/utils/api';

interface CompareSelectModalProps {
  isOpen: boolean;
  onClose: () => void;
  mode: 'dataset' | 'gallery';
  items: { label: string; value: string }[];
  onCompare: (leftValue: string, rightValue: string) => void;
}

function getBasename(filePath: string): string {
  const parts = filePath.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1];
}

const CompareSelectModal: React.FC<CompareSelectModalProps> = ({
  isOpen,
  onClose,
  mode,
  items,
  onCompare,
}) => {
  const [left, setLeft] = useState('');
  const [right, setRight] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setError(null);
      setIsValidating(false);
      if (items.length >= 2) {
        setLeft(items[0].value);
        setRight(items[1].value);
      } else if (items.length === 1) {
        setLeft(items[0].value);
        setRight('');
      } else {
        setLeft('');
        setRight('');
      }
    }
  }, [isOpen, items]);

  const fetchImages = async (value: string): Promise<string[]> => {
    if (mode === 'dataset') {
      const res = await apiClient.post('/api/datasets/listImages', { datasetName: value });
      return (res.data.images as { img_path: string }[]).map(i => i.img_path);
    } else {
      const res = await apiClient.get(`/api/gallery/images?folderPath=${encodeURIComponent(value)}`);
      return (res.data.images as { img_path: string }[]).map(i => i.img_path);
    }
  };

  const handleCompare = async () => {
    if (!left || !right) {
      setError('Please select both a left and right item.');
      return;
    }
    if (left === right) {
      setError('Please select two different items to compare.');
      return;
    }

    setIsValidating(true);
    setError(null);

    try {
      const [leftImages, rightImages] = await Promise.all([
        fetchImages(left),
        fetchImages(right),
      ]);

      if (leftImages.length === 0 && rightImages.length === 0) {
        setError('Both selections are empty. Nothing to compare.');
        setIsValidating(false);
        return;
      }

      if (leftImages.length !== rightImages.length) {
        setError(
          `File count mismatch: left has ${leftImages.length} file(s), right has ${rightImages.length} file(s). Both must contain the same number of files.`
        );
        setIsValidating(false);
        return;
      }

      const leftNames = leftImages.map(p => getBasename(p)).sort();
      const rightNames = rightImages.map(p => getBasename(p)).sort();

      const mismatches: string[] = [];
      for (let i = 0; i < leftNames.length; i++) {
        if (leftNames[i] !== rightNames[i]) {
          mismatches.push(leftNames[i]);
          if (mismatches.length >= 5) break;
        }
      }

      if (mismatches.length > 0) {
        const onlyInLeft = leftNames.filter(n => !rightNames.includes(n));
        const onlyInRight = rightNames.filter(n => !leftNames.includes(n));
        let detail = 'Filenames do not match between the two selections.';
        if (onlyInLeft.length > 0) {
          detail += `\nOnly in left: ${onlyInLeft.slice(0, 3).join(', ')}${onlyInLeft.length > 3 ? '...' : ''}`;
        }
        if (onlyInRight.length > 0) {
          detail += `\nOnly in right: ${onlyInRight.slice(0, 3).join(', ')}${onlyInRight.length > 3 ? '...' : ''}`;
        }
        setError(detail);
        setIsValidating(false);
        return;
      }

      onCompare(left, right);
      onClose();
    } catch (err: any) {
      setError(err?.response?.data?.error || 'Failed to validate selections.');
    } finally {
      setIsValidating(false);
    }
  };

  const itemLabel = mode === 'dataset' ? 'Dataset' : 'Folder';
  const title = mode === 'dataset' ? 'Compare Datasets' : 'Compare Folders';

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={title} size="md">
      <div className="space-y-4 text-gray-200">
        {items.length < 2 ? (
          <>
            <p className="text-gray-400 text-sm">
              You need at least two {mode === 'dataset' ? 'datasets' : 'folders'} to compare.
            </p>
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
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Left {itemLabel}
              </label>
              <select
                className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={left}
                onChange={e => { setLeft(e.target.value); setError(null); }}
              >
                {items.map(item => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Right {itemLabel}
              </label>
              <select
                className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={right}
                onChange={e => { setRight(e.target.value); setError(null); }}
              >
                {items.map(item => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>

            {error && (
              <div className="text-red-400 text-sm whitespace-pre-line bg-red-950/30 border border-red-800/50 rounded-md p-3">
                {error}
              </div>
            )}

            <div className="flex justify-end gap-3 pt-2">
              <button
                type="button"
                className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none"
                onClick={onClose}
                disabled={isValidating}
              >
                Cancel
              </button>
              <button
                type="button"
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none disabled:opacity-50"
                disabled={isValidating || !left || !right || left === right}
                onClick={handleCompare}
              >
                {isValidating ? 'Validating...' : 'Compare'}
              </button>
            </div>
          </>
        )}
      </div>
    </Modal>
  );
};

export default CompareSelectModal;
