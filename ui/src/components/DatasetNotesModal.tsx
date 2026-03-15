'use client';

import { useState, useEffect } from 'react';
import { Modal } from '@/components/Modal';
import { apiClient } from '@/utils/api';

interface DatasetNotesModalProps {
  isOpen: boolean;
  onClose: () => void;
  datasetName: string;
}

export default function DatasetNotesModal({ isOpen, onClose, datasetName }: DatasetNotesModalProps) {
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (isOpen && datasetName) {
      setLoading(true);
      apiClient
        .get(`/api/datasets/notes?datasetName=${encodeURIComponent(datasetName)}`)
        .then(res => {
          setNotes(res.data.notes || '');
        })
        .catch(error => {
          console.error('Error fetching notes:', error);
          setNotes('');
        })
        .finally(() => setLoading(false));
    }
  }, [isOpen, datasetName]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await apiClient.post('/api/datasets/notes', { datasetName, notes });
      onClose();
    } catch (error) {
      console.error('Error saving notes:', error);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`Notes — ${datasetName}`} size="md">
      <div className="space-y-4 text-gray-200">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <span className="text-gray-400">Loading notes...</span>
          </div>
        ) : (
          <textarea
            className="w-full h-48 bg-gray-700 text-gray-200 border border-gray-600 rounded-md p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-vertical"
            value={notes}
            onChange={e => setNotes(e.target.value)}
            placeholder="Add notes about this dataset..."
          />
        )}
        <div className="flex justify-end gap-3">
          <button
            type="button"
            className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            type="button"
            className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            onClick={handleSave}
            disabled={loading || saving}
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </Modal>
  );
}
