'use client';

import { useState, useEffect, useCallback } from 'react';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { openConfirm } from '@/components/ConfirmModal';
import { Button } from '@headlessui/react';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { FaRegTrashAlt } from 'react-icons/fa';

interface VideoDownload {
  id: string;
  url: string;
  dataset: string;
  status: 'pending' | 'downloading' | 'completed' | 'failed';
  progress: number;
  error: string;
  filename: string;
  created_at: string;
  updated_at: string;
}

const statusColors: Record<string, string> = {
  pending: 'text-gray-400',
  downloading: 'text-blue-400',
  completed: 'text-green-400',
  failed: 'text-red-400',
};

const statusLabels: Record<string, string> = {
  pending: 'Pending',
  downloading: 'Downloading',
  completed: 'Completed',
  failed: 'Failed',
};

export default function DownloadsPage() {
  const [downloads, setDownloads] = useState<VideoDownload[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [newUrl, setNewUrl] = useState('');
  const [newDataset, setNewDataset] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formError, setFormError] = useState('');
  const { datasets } = useDatasetList();

  const fetchDownloads = useCallback(async () => {
    try {
      const res = await apiClient.get('/api/downloads');
      setDownloads(res.data.downloads);
    } catch (error) {
      console.error('Error fetching downloads:', error);
    }
  }, []);

  useEffect(() => {
    setIsLoading(true);
    fetchDownloads().finally(() => setIsLoading(false));
  }, [fetchDownloads]);

  // Poll while there are active downloads
  useEffect(() => {
    const hasActive = downloads.some(d => d.status === 'pending' || d.status === 'downloading');
    if (!hasActive) return;

    const timer = setInterval(() => {
      fetchDownloads();
    }, 3000);

    return () => clearInterval(timer);
  }, [downloads, fetchDownloads]);

  const handleAddDownload = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError('');

    if (!newUrl.trim()) {
      setFormError('Please enter a URL.');
      return;
    }
    if (!newDataset.trim()) {
      setFormError('Please select or enter a dataset.');
      return;
    }

    setIsSubmitting(true);
    try {
      await apiClient.post('/api/downloads', { url: newUrl.trim(), dataset: newDataset.trim() });
      setNewUrl('');
      setNewDataset('');
      await fetchDownloads();
    } catch (error: any) {
      setFormError(error?.response?.data?.error || 'Failed to add download.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = (download: VideoDownload) => {
    openConfirm({
      title: 'Remove Download',
      message: `Remove this download from the queue?\n${download.url}`,
      type: 'warning',
      confirmText: 'Remove',
      onConfirm: () => {
        apiClient
          .post('/api/downloads/delete', { id: download.id })
          .then(() => fetchDownloads())
          .catch(error => console.error('Error deleting download:', error));
      },
    });
  };

  const tableRows = downloads.map(d => ({ ...d }));

  const columns: TableColumn[] = [
    {
      title: 'URL',
      key: 'url',
      render: row => (
        <a
          href={row.url}
          target="_blank"
          rel="noreferrer"
          className="text-blue-400 hover:text-blue-300 truncate max-w-xs block"
          title={row.url}
        >
          {row.url.length > 60 ? row.url.slice(0, 60) + '…' : row.url}
        </a>
      ),
    },
    {
      title: 'Dataset',
      key: 'dataset',
      className: 'w-36',
      render: row => <span className="text-gray-300">{row.dataset}</span>,
    },
    {
      title: 'Status',
      key: 'status',
      className: 'w-32',
      render: row => (
        <div>
          <span className={statusColors[row.status] ?? 'text-gray-400'}>
            {statusLabels[row.status] ?? row.status}
          </span>
          {row.status === 'failed' && row.error && (
            <div className="text-xs text-red-400 mt-1 truncate max-w-xs" title={row.error}>
              {row.error}
            </div>
          )}
        </div>
      ),
    },
    {
      title: 'Progress',
      key: 'progress',
      className: 'w-48',
      render: row => {
        if (row.status === 'completed') {
          return (
            <div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '100%' }} />
              </div>
              {row.filename && (
                <div className="text-xs text-gray-400 mt-1 truncate" title={row.filename}>
                  {row.filename}
                </div>
              )}
            </div>
          );
        }
        if (row.status === 'downloading' || row.status === 'pending') {
          return (
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${row.progress}%` }}
              />
            </div>
          );
        }
        if (row.status === 'failed') {
          return <span className="text-red-400 text-xs">—</span>;
        }
        return null;
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-16 text-right',
      render: row => (
        <button
          className="text-gray-200 hover:bg-red-600 p-2 rounded-full transition-colors"
          onClick={() => handleDelete(row)}
          disabled={row.status === 'downloading'}
          title={row.status === 'downloading' ? 'Cannot remove an active download' : 'Remove'}
        >
          <FaRegTrashAlt className={row.status === 'downloading' ? 'opacity-30' : ''} />
        </button>
      ),
    },
  ];

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-2xl font-semibold text-gray-100">Downloads</h1>
        </div>
      </TopBar>

      <MainContent>
        {/* Add download form */}
        <div className="mb-6 bg-gray-900 rounded-md p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">Add Download</h2>
          <form onSubmit={handleAddDownload} className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <div className="flex-1">
              <TextInput
                label="Video URL"
                value={newUrl}
                onChange={setNewUrl}
                placeholder="https://www.youtube.com/watch?v=..."
              />
            </div>
            <div className="w-full sm:w-56">
              <label className="block text-xs mb-1 mt-2 text-gray-300">Dataset</label>
              <input
                list="dataset-list"
                value={newDataset}
                onChange={e => setNewDataset(e.target.value)}
                placeholder="Select or type dataset name"
                className="w-full text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent"
              />
              <datalist id="dataset-list">
                {datasets.map(d => (
                  <option key={d} value={d} />
                ))}
              </datalist>
            </div>
            <div className="sm:pb-0.5">
              <Button
                type="submit"
                disabled={isSubmitting}
                className="text-gray-200 bg-slate-600 px-4 py-1.5 rounded-md hover:bg-slate-500 transition-colors disabled:opacity-50"
              >
                {isSubmitting ? 'Adding…' : 'Add'}
              </Button>
            </div>
          </form>
          {formError && <p className="mt-2 text-sm text-red-400">{formError}</p>}
        </div>

        {/* Downloads table */}
        <UniversalTable
          columns={columns}
          rows={tableRows}
          isLoading={isLoading}
          onRefresh={fetchDownloads}
        />
      </MainContent>
    </>
  );
}
