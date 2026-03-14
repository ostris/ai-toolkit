'use client';

import React, { useState, useEffect } from 'react';
import { Modal } from './Modal';
import { apiClient } from '@/utils/api';

interface CompareSelectModalProps {
  isOpen: boolean;
  onClose: () => void;
  mode: 'dataset' | 'gallery';
  items: { label: string; value: string }[];
  onCompare: (leftValue: string, rightValue: string, centerValue?: string) => void;
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
  const [center, setCenter] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setError(null);
      setIsValidating(false);
      setCenter('');
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
    const selected = [left, right, ...(center ? [center] : [])];
    const uniqueSelected = new Set(selected);
    if (uniqueSelected.size !== selected.length) {
      setError('Please select different items to compare.');
      return;
    }

    setIsValidating(true);
    setError(null);

    try {
      const fetchPromises = selected.map(v => fetchImages(v));
      const allImages = await Promise.all(fetchPromises);

      if (allImages.every(imgs => imgs.length === 0)) {
        setError('All selections are empty. Nothing to compare.');
        setIsValidating(false);
        return;
      }

      const nameSets = allImages.map(imgs => new Set(imgs.map(p => getBasename(p))));

      // Find the overlap — files present in all selected datasets
      const commonNames = [...nameSets[0]].filter(n => nameSets.every(s => s.has(n)));

      if (commonNames.length === 0) {
        setError('No matching filenames found across all selections.');
        setIsValidating(false);
        return;
      }

      // Allow if one set is a superset of all others (or they're identical)
      const hasSuperset = nameSets.some(superSet =>
        nameSets.every(otherSet => [...otherSet].every(n => superSet.has(n)))
      );

      if (!hasSuperset) {
        setError('One selection must contain all files of the others. No single selection is a superset of the rest.');
        setIsValidating(false);
        return;
      }

      onCompare(left, right, center || undefined);
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

            {items.length >= 3 && (
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Center {itemLabel} <span className="text-gray-500 font-normal">(optional)</span>
                </label>
                <select
                  className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={center}
                  onChange={e => { setCenter(e.target.value); setError(null); }}
                >
                  <option value="">— None —</option>
                  {items.map(item => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

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
                disabled={isValidating || !left || !right || left === right || (!!center && (center === left || center === right))}
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
