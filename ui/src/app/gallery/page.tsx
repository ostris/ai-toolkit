'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt, FaInfoCircle, FaFolderPlus } from 'react-icons/fa';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { Tooltip } from '@/components/Tooltip';
import { formatDuration } from '@/utils/basic';
import { Modal } from '@/components/Modal';

interface GalleryFolder {
  id: number;
  path: string;
  created_at: string;
}

interface ImageStats {
  totalCount: number;
  imageCount: number;
  videoCount: number;
  totalVideoDuration: number;
  resolutionBreakdown: { [resolution: string]: number };
  error?: boolean;
}

export default function GalleryPage() {
  const [folders, setFolders] = useState<GalleryFolder[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [imageStats, setImageStats] = useState<{ [folderPath: string]: ImageStats }>({});
  const [statsLoading, setStatsLoading] = useState<{ [folderPath: string]: boolean }>({});
  const requestedFolders = useRef<Set<string>>(new Set());
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [newFolderPath, setNewFolderPath] = useState('');
  const [addRecursive, setAddRecursive] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);
  const [isAdding, setIsAdding] = useState(false);

  const refreshFolders = () => {
    setStatus('loading');
    apiClient
      .get('/api/gallery/list')
      .then(res => res.data)
      .then((data: GalleryFolder[]) => {
        setFolders(data);
        setStatus('success');
      })
      .catch(() => {
        setStatus('error');
      });
  };

  useEffect(() => {
    refreshFolders();
  }, []);

  useEffect(() => {
    const abortController = new AbortController();
    if (folders.length > 0) {
      folders.forEach(folder => {
        if (!requestedFolders.current.has(folder.path)) {
          requestedFolders.current.add(folder.path);
          setStatsLoading(prev => ({ ...prev, [folder.path]: true }));
          apiClient
            .get(`/api/gallery/imageStats?folderPath=${encodeURIComponent(folder.path)}`, { signal: abortController.signal })
            .then(res => res.data)
            .then((data: ImageStats) => {
              if (!abortController.signal.aborted) {
                setImageStats(prev => ({ ...prev, [folder.path]: data }));
                setStatsLoading(prev => ({ ...prev, [folder.path]: false }));
              }
            })
            .catch(error => {
              if (!abortController.signal.aborted) {
                console.error(`Error fetching stats for ${folder.path}:`, error);
                setImageStats(prev => ({
                  ...prev,
                  [folder.path]: { totalCount: 0, imageCount: 0, videoCount: 0, totalVideoDuration: 0, resolutionBreakdown: {}, error: true },
                }));
                setStatsLoading(prev => ({ ...prev, [folder.path]: false }));
              }
            });
        }
      });
    }
    return () => {
      abortController.abort();
    };
  }, [folders]);

  const handleRemoveFolder = (folder: GalleryFolder) => {
    openConfirm({
      title: 'Remove Gallery Folder',
      message: `Remove "${folder.path}" from the gallery? The folder and its files will not be deleted.`,
      type: 'warning',
      confirmText: 'Remove',
      onConfirm: () => {
        apiClient
          .post('/api/gallery/remove', { id: folder.id })
          .then(() => {
            setImageStats(prev => {
              const next = { ...prev };
              delete next[folder.path];
              return next;
            });
            setStatsLoading(prev => {
              const next = { ...prev };
              delete next[folder.path];
              return next;
            });
            requestedFolders.current.delete(folder.path);
            refreshFolders();
          })
          .catch(err => {
            console.error('Error removing gallery folder:', err);
          });
      },
    });
  };

  const handleAddFolder = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newFolderPath.trim()) return;
    setIsAdding(true);
    setAddError(null);
    try {
      await apiClient.post('/api/gallery/add', { folderPath: newFolderPath.trim(), recursive: addRecursive });
      setNewFolderPath('');
      setAddRecursive(false);
      setIsAddModalOpen(false);
      refreshFolders();
    } catch (err: any) {
      setAddError(err?.response?.data?.error || 'Failed to add folder');
    } finally {
      setIsAdding(false);
    }
  };

  const tableRows = folders.map(folder => ({
    id: folder.id,
    path: folder.path,
    stats: folder.path,
    videoStats: folder.path,
    actions: folder,
  }));

  const columns: TableColumn[] = [
    {
      title: 'Folder Path',
      key: 'path',
      render: row => (
        <Link href={`/gallery/${row.id}`} className="text-gray-200 hover:text-gray-100 break-all">
          {row.path}
        </Link>
      ),
    },
    {
      title: 'Image Count',
      key: 'stats',
      className: 'w-32 text-center',
      render: row => {
        const stats = imageStats[row.path];
        const loading = statsLoading[row.path];
        if (loading) return <span className="text-gray-400">Loading...</span>;
        if (!stats) return <span className="text-gray-400">-</span>;
        if (stats.error && stats.totalCount === 0) return <span className="text-red-400">Error</span>;

        const sortedResolutions = Object.entries(stats.resolutionBreakdown).sort(([, a], [, b]) => b - a);
        const tooltipContent = (
          <div className="text-left">
            <div className="font-semibold mb-1">Resolution Breakdown:</div>
            {sortedResolutions.length > 0 ? (
              sortedResolutions.map(([res, count]) => (
                <div key={res} className="text-xs">{res}: {count} {count === 1 ? 'image' : 'images'}</div>
              ))
            ) : (
              <div className="text-xs">No images found</div>
            )}
          </div>
        );

        return (
          <div className="flex items-center justify-center gap-2">
            <span className="text-gray-200">{stats.imageCount}</span>
            <Tooltip content={tooltipContent}>
              <FaInfoCircle className="text-gray-400 hover:text-gray-200 cursor-help" />
            </Tooltip>
          </div>
        );
      },
    },
    {
      title: 'Video Count',
      key: 'videoStats',
      className: 'w-40 text-center',
      render: row => {
        const stats = imageStats[row.path];
        const loading = statsLoading[row.path];
        if (loading) return <span className="text-gray-400">Loading...</span>;
        if (!stats) return <span className="text-gray-400">-</span>;
        if (stats.error) return <span className="text-gray-400">-</span>;
        if (stats.videoCount === 0) return <span className="text-gray-400">0</span>;
        return <span className="text-gray-200">{stats.videoCount} ({formatDuration(stats.totalVideoDuration)})</span>;
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-20 text-right',
      render: row => (
        <button
          className="text-gray-200 hover:bg-red-600 p-2 rounded-full transition-colors"
          onClick={() => handleRemoveFolder(row.actions)}
        >
          <FaRegTrashAlt />
        </button>
      ),
    },
  ];

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-2xl font-semibold text-gray-100">Gallery</h1>
        </div>
        <div className="flex-1" />
        <div>
          <Button
            className="text-gray-200 bg-slate-600 px-4 py-2 rounded-md hover:bg-slate-500 transition-colors flex items-center gap-2"
            onClick={() => { setAddError(null); setIsAddModalOpen(true); }}
          >
            <FaFolderPlus />
            Add Folder
          </Button>
        </div>
      </TopBar>

      <MainContent>
        <UniversalTable
          columns={columns}
          rows={tableRows}
          isLoading={status === 'loading'}
          onRefresh={refreshFolders}
        />
      </MainContent>

      <Modal isOpen={isAddModalOpen} onClose={() => setIsAddModalOpen(false)} title="Add Gallery Folder" size="md">
        <div className="space-y-4 text-gray-200">
          <form onSubmit={handleAddFolder}>
            <div className="text-sm text-gray-400">
              Enter the absolute path to a folder you want to add to the gallery.
            </div>
            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-300 mb-1">Folder Path</label>
              <input
                type="text"
                value={newFolderPath}
                onChange={e => setNewFolderPath(e.target.value)}
                placeholder="/path/to/your/photos"
                className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                autoFocus
              />
            </div>
            <div className="mt-3 flex items-center gap-2">
              <input
                type="checkbox"
                id="recursive"
                checked={addRecursive}
                onChange={e => setAddRecursive(e.target.checked)}
                className="accent-blue-500"
              />
              <label htmlFor="recursive" className="text-sm text-gray-300 cursor-pointer">
                Also add all subfolders recursively
              </label>
            </div>
            {addError && <p className="mt-2 text-red-400 text-sm">{addError}</p>}
            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none"
                onClick={() => setIsAddModalOpen(false)}
                disabled={isAdding}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none disabled:opacity-50"
                disabled={isAdding || !newFolderPath.trim()}
              >
                {isAdding ? 'Adding...' : 'Add'}
              </button>
            </div>
          </form>
        </div>
      </Modal>
    </>
  );
}
