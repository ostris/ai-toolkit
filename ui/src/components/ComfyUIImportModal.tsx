'use client';

import { useState, useEffect } from 'react';
import { Modal } from '@/components/Modal';
import { apiClient } from '@/utils/api';

interface ComfyUIImportModalProps {
  isOpen: boolean;
  onClose: () => void;
  onImportComplete: () => void;
}

export default function ComfyUIImportModal({ isOpen, onClose, onImportComplete }: ComfyUIImportModalProps) {
  const [folders, setFolders] = useState<string[]>([]);
  const [selectedFolder, setSelectedFolder] = useState('');
  const [datasetName, setDatasetName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [foldersLoading, setFoldersLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setError('');
      setDatasetName('');
      setSelectedFolder('');
      setFoldersLoading(true);
      apiClient
        .get('/api/comfyui/list-output-folders')
        .then(res => {
          setFolders(res.data.folders);
          if (res.data.folders.length > 0) {
            setSelectedFolder(res.data.folders[0]);
          }
        })
        .catch(err => {
          setError(err?.response?.data?.error || 'Failed to load ComfyUI output folders. Check your settings.');
          setFolders([]);
        })
        .finally(() => setFoldersLoading(false));
    }
  }, [isOpen]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!datasetName.trim() || !selectedFolder) return;

    setLoading(true);
    setError('');

    try {
      await apiClient.post('/api/comfyui/import', {
        name: datasetName,
        folder: selectedFolder,
      });
      onImportComplete();
      onClose();
    } catch (err: any) {
      setError(err?.response?.data?.error || 'Failed to import from ComfyUI');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Import from ComfyUI" size="md">
      <div className="space-y-4 text-gray-200">
        {error && (
          <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded-md px-3 py-2">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label htmlFor="comfyui-dataset-name" className="block text-sm font-medium mb-2">
                Dataset Name
              </label>
              <input
                type="text"
                id="comfyui-dataset-name"
                value={datasetName}
                onChange={e => setDatasetName(e.target.value)}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                placeholder="Enter dataset name"
                required
              />
            </div>

            <div>
              <label htmlFor="comfyui-source-folder" className="block text-sm font-medium mb-2">
                Source Folder
              </label>
              {foldersLoading ? (
                <p className="text-gray-400 text-sm">Loading folders...</p>
              ) : (
                <select
                  id="comfyui-source-folder"
                  value={selectedFolder}
                  onChange={e => setSelectedFolder(e.target.value)}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                  disabled={folders.length === 0}
                >
                  {folders.map(folder => (
                    <option key={folder} value={folder}>
                      {folder}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </div>

          <div className="mt-6 flex justify-end space-x-3">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
              onClick={onClose}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !datasetName.trim() || !selectedFolder}
              className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Importing...' : 'Import'}
            </button>
          </div>
        </form>
      </div>
    </Modal>
  );
}
