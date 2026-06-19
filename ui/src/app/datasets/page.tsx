'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { Button } from '@headlessui/react';
import { MoreVertical, Pencil, Copy, Trash2, Image as ImageIcon } from 'lucide-react';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';
import DatasetThumbnailPager from './DatasetThumbnailPager';

interface DatasetStats {
  name: string;
  image_count: number;
  total_size: number;
  modified_at: number;
  thumbs?: string[];
}

const THUMBNAILS_STORAGE_KEY = 'AITK_DATASETS_SHOW_THUMBS';

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let value = bytes;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  const fixed = value >= 100 || i === 0 ? value.toFixed(0) : value.toFixed(1);
  return `${fixed} ${units[i]}`;
}

export default function Datasets() {
  const router = useRouter();
  const { datasets, status, refreshDatasets } = useDatasetList();
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [stats, setStats] = useState<Record<string, DatasetStats>>({});
  const [statsLoading, setStatsLoading] = useState(false);
  const [menuOpenFor, setMenuOpenFor] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const [showThumbs, setShowThumbs] = useState(false);

  useEffect(() => {
    try {
      setShowThumbs(localStorage.getItem(THUMBNAILS_STORAGE_KEY) === '1');
    } catch {}
  }, []);

  const toggleThumbs = () => {
    setShowThumbs(prev => {
      const next = !prev;
      try {
        localStorage.setItem(THUMBNAILS_STORAGE_KEY, next ? '1' : '0');
      } catch {}
      return next;
    });
  };

  const refreshStats = () => {
    setStatsLoading(true);
    apiClient
      .get('/api/datasets/stats')
      .then(res => {
        const map: Record<string, DatasetStats> = {};
        (res.data?.datasets || []).forEach((d: DatasetStats) => {
          map[d.name] = d;
        });
        setStats(map);
      })
      .catch(err => console.error('Error loading dataset stats:', err))
      .finally(() => setStatsLoading(false));
  };

  useEffect(() => {
    refreshStats();
  }, []);

  // Close the action menu on any outside click.
  useEffect(() => {
    if (!menuOpenFor) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpenFor(null);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [menuOpenFor]);

  const refreshAll = () => {
    refreshDatasets();
    refreshStats();
  };

  const tableRows = useMemo(
    () =>
      datasets.map(dataset => ({
        name: dataset,
        image_count: stats[dataset]?.image_count,
        total_size: stats[dataset]?.total_size,
        thumbs: stats[dataset]?.thumbs || [],
      })),
    [datasets, stats],
  );

  const handleDeleteDataset = (datasetName: string) => {
    setMenuOpenFor(null);
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/datasets/delete', { name: datasetName })
          .then(() => {
            refreshAll();
          })
          .catch(err => console.error('Error deleting dataset:', err));
      },
    });
  };

  const handleRenameDataset = (datasetName: string) => {
    setMenuOpenFor(null);
    openConfirm({
      title: 'Rename Dataset',
      message: `Enter a new name for "${datasetName}". Names are lower-cased; spaces and special characters become underscores.`,
      type: 'info',
      confirmText: 'Rename',
      inputTitle: 'New Name',
      onConfirm: async (newName?: string) => {
        if (!newName) return;
        try {
          await apiClient.post('/api/datasets/rename', { name: datasetName, newName });
          refreshAll();
        } catch (err: any) {
          alert(err.response?.data?.error || 'Failed to rename dataset.');
        }
      },
    });
  };

  const handleCloneDataset = (datasetName: string) => {
    setMenuOpenFor(null);
    openConfirm({
      title: 'Clone Dataset',
      message: `Copy all images and captions from "${datasetName}" into a new dataset. Leave the name blank to use "${datasetName}_copy".`,
      type: 'info',
      confirmText: 'Clone',
      inputTitle: 'New Dataset Name (optional)',
      onConfirm: async (newName?: string) => {
        try {
          const res = await apiClient.post('/api/datasets/clone', {
            name: datasetName,
            newName: newName || `${datasetName}_copy`,
          });
          refreshAll();
          if (res.data?.name) {
            router.push(`/datasets/${res.data.name}`);
          }
        } catch (err: any) {
          alert(err.response?.data?.error || 'Failed to clone dataset.');
        }
      },
    });
  };

  const thumbColumn: TableColumn = {
    title: 'Preview',
    key: 'thumbs',
    // No fixed width — the column expands to absorb the dead space between
    // "Dataset Name" and "Images".
    render: row => (
      <DatasetThumbnailPager
        datasetName={row.name}
        initialThumbs={row.thumbs || []}
        statsLoading={statsLoading}
      />
    ),
  };

  const columns: TableColumn[] = [
    {
      title: 'Dataset Name',
      key: 'name',
      className: showThumbs ? 'w-56 align-top' : undefined,
      render: row => (
        <Link href={`/datasets/${row.name}`} className="text-gray-200 hover:text-gray-100">
          {row.name}
        </Link>
      ),
    },
    ...(showThumbs ? [thumbColumn] : []),
    {
      title: 'Images',
      key: 'image_count',
      className: 'w-28 text-right',
      render: row =>
        row.image_count === undefined ? (
          <span className="text-gray-500">{statsLoading ? '…' : '—'}</span>
        ) : (
          <span className="text-gray-300 tabular-nums">{row.image_count.toLocaleString()}</span>
        ),
    },
    {
      title: 'Total Size',
      key: 'total_size',
      className: 'w-32 text-right',
      render: row =>
        row.total_size === undefined ? (
          <span className="text-gray-500">{statsLoading ? '…' : '—'}</span>
        ) : (
          <span className="text-gray-300 tabular-nums">{formatBytes(row.total_size)}</span>
        ),
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-16 text-right',
      render: row => {
        const isOpen = menuOpenFor === row.name;
        return (
          <div className="relative inline-block" ref={isOpen ? menuRef : undefined}>
            <button
              className="text-gray-300 hover:text-white hover:bg-gray-800 p-2 rounded-full transition-colors"
              onClick={e => {
                e.stopPropagation();
                setMenuOpenFor(prev => (prev === row.name ? null : row.name));
              }}
              title="Actions"
            >
              <MoreVertical className="w-4 h-4" />
            </button>
            {isOpen && (
              <div className="absolute right-0 top-full mt-1 z-20 w-40 bg-gray-900 border border-gray-700 rounded-md shadow-lg py-1">
                <button
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-800 text-left"
                  onClick={() => handleRenameDataset(row.name)}
                >
                  <Pencil className="w-4 h-4" /> Rename
                </button>
                <button
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-800 text-left"
                  onClick={() => handleCloneDataset(row.name)}
                >
                  <Copy className="w-4 h-4" /> Clone
                </button>
                <div className="border-t border-gray-800 my-1" />
                <button
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-red-400 hover:bg-red-950/40 text-left"
                  onClick={() => handleDeleteDataset(row.name)}
                >
                  <Trash2 className="w-4 h-4" /> Delete
                </button>
              </div>
            )}
          </div>
        );
      },
    },
  ];

  const handleCreateDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const data = await apiClient.post('/api/datasets/create', { name: newDatasetName }).then(res => res.data);
      refreshAll();
      setNewDatasetName('');
      setIsNewDatasetModalOpen(false);
    } catch (error) {
      console.error('Error creating new dataset:', error);
    }
  };

  const openNewDatasetModal = () => {
    openConfirm({
      title: 'New Dataset',
      message: 'Enter the name of the new dataset:',
      type: 'info',
      confirmText: 'Create',
      inputTitle: 'Dataset Name',
      onConfirm: async (name?: string) => {
        if (!name) return;
        try {
          const data = await apiClient.post('/api/datasets/create', { name }).then(res => res.data);
          if (data.name) {
            router.push(`/datasets/${data.name}`);
          } else {
            refreshAll();
          }
        } catch (error) {
          console.error('Error creating new dataset:', error);
        }
      },
    });
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Datasets</h1>
        </div>
        <div className="flex-1"></div>
        <div className="flex items-center gap-2 pr-2">
          <Button
            onClick={toggleThumbs}
            title={showThumbs ? 'Hide thumbnail previews' : 'Show thumbnail previews'}
            className={`flex items-center gap-1 px-3 py-1 rounded-md text-sm transition-colors ${
              showThumbs
                ? 'text-white bg-blue-600 hover:bg-blue-500'
                : 'text-gray-200 bg-gray-800 hover:bg-gray-700'
            }`}
          >
            <ImageIcon className="w-4 h-4" />
            {showThumbs ? 'Hide Thumbnails' : 'Show Thumbnails'}
          </Button>
          <Button
            className="text-white bg-slate-600 px-2 sm:px-3 py-1 rounded-md hover:bg-slate-500 transition-colors text-sm sm:text-base whitespace-nowrap"
            onClick={() => openNewDatasetModal()}
          >
            <span className="sm:hidden">+ New</span>
            <span className="hidden sm:inline">New Dataset</span>
          </Button>
        </div>
      </TopBar>

      <MainContent>
        <UniversalTable
          columns={columns}
          rows={tableRows}
          isLoading={status === 'loading'}
          onRefresh={refreshAll}
        />
      </MainContent>

      <Modal
        isOpen={isNewDatasetModalOpen}
        onClose={() => setIsNewDatasetModalOpen(false)}
        title="New Dataset"
        size="md"
      >
        <div className="space-y-4 text-gray-200">
          <form onSubmit={handleCreateDataset}>
            <div className="text-sm text-gray-400">
              This will create a new folder with the name below in your dataset folder.
            </div>
            <div className="mt-4">
              <TextInput label="Dataset Name" value={newDatasetName} onChange={value => setNewDatasetName(value)} />
            </div>

            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
                onClick={() => setIsNewDatasetModalOpen(false)}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                Confirm
              </button>
            </div>
          </form>
        </div>
      </Modal>
    </>
  );
}
