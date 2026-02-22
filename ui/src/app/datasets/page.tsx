'use client';

import { useState, useEffect } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt, FaInfoCircle } from 'react-icons/fa';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';
import { Tooltip } from '@/components/Tooltip';

interface ImageStats {
  totalCount: number;
  resolutionBreakdown: { [resolution: string]: number };
  error?: boolean; // Flag to indicate if there was an error
}

export default function Datasets() {
  const router = useRouter();
  const { datasets, status, refreshDatasets } = useDatasetList();
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [imageStats, setImageStats] = useState<{ [datasetName: string]: ImageStats }>({});
  const [statsLoading, setStatsLoading] = useState<{ [datasetName: string]: boolean }>({});

  // Fetch image stats for each dataset
  useEffect(() => {
    const abortController = new AbortController();
    
    if (datasets.length > 0) {
      datasets.forEach(datasetName => {
        // Only fetch if we don't already have the stats and not currently loading
        if (!imageStats[datasetName] && !statsLoading[datasetName]) {
          setStatsLoading(prev => ({ ...prev, [datasetName]: true }));
          apiClient
            .post('/api/datasets/imageStats', { datasetName }, { signal: abortController.signal })
            .then(res => res.data)
            .then((data: ImageStats) => {
              if (!abortController.signal.aborted) {
                setImageStats(prev => ({ ...prev, [datasetName]: data }));
                setStatsLoading(prev => ({ ...prev, [datasetName]: false }));
              }
            })
            .catch(error => {
              if (!abortController.signal.aborted) {
                console.error(`Error fetching image stats for ${datasetName}:`, error);
                // Set error state so we can show "error fetching stats"
                setImageStats(prev => ({ 
                  ...prev, 
                  [datasetName]: { totalCount: 0, resolutionBreakdown: {}, error: true } 
                }));
                setStatsLoading(prev => ({ ...prev, [datasetName]: false }));
              }
            });
        }
      });
    }

    return () => {
      abortController.abort();
    };
  }, [datasets, imageStats, statsLoading]);

  // Transform datasets array into rows with objects
  const tableRows = datasets.map(dataset => ({
    name: dataset,
    imageCount: dataset, // Pass dataset name to look up stats
    actions: dataset, // Pass full dataset name for actions
  }));

  const columns: TableColumn[] = [
    {
      title: 'Dataset Name',
      key: 'name',
      render: row => (
        <Link href={`/datasets/${row.name}`} className="text-gray-200 hover:text-gray-100">
          {row.name}
        </Link>
      ),
    },
    {
      title: 'Image Count',
      key: 'imageCount',
      className: 'w-32 text-center',
      render: row => {
        const datasetName = row.name;
        const stats = imageStats[datasetName];
        const loading = statsLoading[datasetName];

        if (loading) {
          return <span className="text-gray-400">Loading...</span>;
        }

        if (!stats) {
          return <span className="text-gray-400">-</span>;
        }

        // If there was an error and no data, show error message
        if (stats.error && stats.totalCount === 0) {
          return <span className="text-red-400">Error fetching stats</span>;
        }

        // Sort resolutions by count (descending)
        const sortedResolutions = Object.entries(stats.resolutionBreakdown).sort(
          ([, countA], [, countB]) => countB - countA
        );

        const tooltipContent = (
          <div className="text-left">
            <div className="font-semibold mb-1">Resolution Breakdown:</div>
            {sortedResolutions.length > 0 ? (
              sortedResolutions.map(([resolution, count]) => (
                <div key={resolution} className="text-xs">
                  {resolution}: {count} {count === 1 ? 'image' : 'images'}
                </div>
              ))
            ) : (
              <div className="text-xs">No images found</div>
            )}
          </div>
        );

        return (
          <div className="flex items-center justify-center gap-2">
            <span className="text-gray-200">{stats.totalCount}</span>
            <Tooltip content={tooltipContent}>
              <FaInfoCircle className="text-gray-400 hover:text-gray-200 cursor-help" />
            </Tooltip>
          </div>
        );
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-20 text-right',
      render: row => (
        <button
          className="text-gray-200 hover:bg-red-600 p-2 rounded-full transition-colors"
          onClick={() => handleDeleteDataset(row.name)}
        >
          <FaRegTrashAlt />
        </button>
      ),
    },
  ];

  const handleDeleteDataset = (datasetName: string) => {
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/datasets/delete', { name: datasetName })
          .then(() => {
            console.log('Dataset deleted:', datasetName);
            // Clear stats for the deleted dataset
            setImageStats(prev => {
              const newStats = { ...prev };
              delete newStats[datasetName];
              return newStats;
            });
            setStatsLoading(prev => {
              const newLoading = { ...prev };
              delete newLoading[datasetName];
              return newLoading;
            });
            refreshDatasets();
          })
          .catch(error => {
            console.error('Error deleting dataset:', error);
          });
      },
    });
  };

  const handleCreateDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const data = await apiClient.post('/api/datasets/create', { name: newDatasetName }).then(res => res.data);
      console.log('New dataset created:', data);
      refreshDatasets();
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
        if (!name) {
          console.error('Dataset name is required.');
          return;
        }
        try {
          const data = await apiClient.post('/api/datasets/create', { name }).then(res => res.data);
          console.log('New dataset created:', data);
          if (data.name) {
            router.push(`/datasets/${data.name}`);
          } else {
            refreshDatasets();
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
          <h1 className="text-2xl font-semibold text-gray-100">Datasets</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-gray-200 bg-slate-600 px-4 py-2 rounded-md hover:bg-slate-500 transition-colors"
            onClick={() => openNewDatasetModal()}
          >
            New Dataset
          </Button>
        </div>
      </TopBar>

      <MainContent>
        <UniversalTable
          columns={columns}
          rows={tableRows}
          isLoading={status === 'loading'}
          onRefresh={refreshDatasets}
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
