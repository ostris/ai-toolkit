'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { openConfirm } from '@/components/ConfirmModal';
import { Button } from '@headlessui/react';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { FaRegTrashAlt, FaChevronDown, FaChevronUp } from 'react-icons/fa';
import { VideoInfo } from '../api/downloads/info/route';

interface VideoDownload {
  id: string;
  url: string;
  dataset: string;
  status: 'pending' | 'downloading' | 'completed' | 'failed';
  progress: number;
  error: string;
  filename: string;
  title: string;
  filesize: string;
  speed: string;
  format: string;
  cookies_file: string;
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

function formatDuration(seconds: number | null): string {
  if (!seconds) return '';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

export default function DownloadsPage() {
  const [downloads, setDownloads] = useState<VideoDownload[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Form state
  const [newUrl, setNewUrl] = useState('');
  const [newDataset, setNewDataset] = useState('');
  const [newFormat, setNewFormat] = useState('');
  const [newCookiesFile, setNewCookiesFile] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formError, setFormError] = useState('');

  // Metadata fetch state
  const [isFetching, setIsFetching] = useState(false);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [infoError, setInfoError] = useState('');
  const fetchedUrlRef = useRef('');

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

  const handleFetchInfo = async () => {
    if (!newUrl.trim()) {
      setInfoError('Please enter a URL first.');
      return;
    }
    if (fetchedUrlRef.current === newUrl.trim()) return; // already fetched

    setIsFetching(true);
    setVideoInfo(null);
    setInfoError('');
    setNewFormat('');

    try {
      const res = await apiClient.get(`/api/downloads/info?url=${encodeURIComponent(newUrl.trim())}`);
      setVideoInfo(res.data);
      fetchedUrlRef.current = newUrl.trim();
      // Default to best quality
      if (res.data.resolutions?.length > 0) {
        setNewFormat(res.data.resolutions[0].format);
      }
    } catch (error: any) {
      setInfoError(error?.response?.data?.error || 'Failed to fetch video info.');
    } finally {
      setIsFetching(false);
    }
  };

  const handleUrlChange = (val: string) => {
    setNewUrl(val);
    if (val.trim() !== fetchedUrlRef.current) {
      setVideoInfo(null);
      setInfoError('');
      setNewFormat('');
    }
  };

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
      await apiClient.post('/api/downloads', {
        url: newUrl.trim(),
        dataset: newDataset.trim(),
        format: newFormat.trim(),
        title: videoInfo?.title ?? '',
        cookies_file: newCookiesFile.trim(),
      });
      setNewUrl('');
      setNewDataset('');
      setNewFormat('');
      setNewCookiesFile('');
      setVideoInfo(null);
      fetchedUrlRef.current = '';
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
      title: 'Video',
      key: 'url',
      render: row => (
        <div>
          {row.title && <div className="text-gray-200 text-sm mb-0.5 truncate max-w-xs" title={row.title}>{row.title}</div>}
          <a
            href={row.url}
            target="_blank"
            rel="noreferrer"
            className="text-blue-400 hover:text-blue-300 text-xs truncate max-w-xs block"
            title={row.url}
          >
            {row.url.length > 60 ? row.url.slice(0, 60) + '…' : row.url}
          </a>
        </div>
      ),
    },
    {
      title: 'Dataset',
      key: 'dataset',
      className: 'w-32',
      render: row => <span className="text-gray-300 text-sm">{row.dataset}</span>,
    },
    {
      title: 'Status',
      key: 'status',
      className: 'w-28',
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
      className: 'w-56',
      render: row => {
        if (row.status === 'completed') {
          return (
            <div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '100%' }} />
              </div>
              {row.filesize && (
                <div className="text-xs text-gray-400 mt-1">{row.filesize}</div>
              )}
              {row.filename && (
                <div className="text-xs text-gray-500 mt-0.5 truncate" title={row.filename}>
                  {row.filename}
                </div>
              )}
            </div>
          );
        }
        if (row.status === 'downloading') {
          return (
            <div>
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>{row.progress.toFixed(1)}%</span>
                {row.speed && <span>{row.speed}</span>}
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${row.progress}%` }}
                />
              </div>
              {row.filesize && (
                <div className="text-xs text-gray-400 mt-1">{row.filesize}</div>
              )}
            </div>
          );
        }
        if (row.status === 'pending') {
          return (
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div className="bg-gray-600 h-2 rounded-full" style={{ width: '0%' }} />
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
      className: 'w-14 text-right',
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
          <form onSubmit={handleAddDownload}>
            {/* Row 1: URL + Fetch button */}
            <div className="flex gap-2 items-end">
              <div className="flex-1">
                <TextInput
                  label="Video URL"
                  value={newUrl}
                  onChange={handleUrlChange}
                  placeholder="https://www.youtube.com/watch?v=..."
                />
              </div>
              <div className="pb-0.5">
                <Button
                  type="button"
                  onClick={handleFetchInfo}
                  disabled={isFetching || !newUrl.trim()}
                  className="text-gray-200 bg-gray-700 px-3 py-1.5 rounded-md hover:bg-gray-600 transition-colors disabled:opacity-50 whitespace-nowrap"
                >
                  {isFetching ? 'Fetching…' : 'Fetch Info'}
                </Button>
              </div>
            </div>

            {infoError && <p className="mt-1 text-sm text-red-400">{infoError}</p>}

            {/* Video preview */}
            {videoInfo && (
              <div className="mt-3 flex gap-3 bg-gray-800 rounded-md p-3">
                {videoInfo.thumbnail && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={videoInfo.thumbnail}
                    alt={videoInfo.title}
                    className="w-28 h-16 object-cover rounded flex-shrink-0"
                  />
                )}
                <div className="flex-1 min-w-0">
                  <div className="text-gray-100 text-sm font-medium truncate">{videoInfo.title}</div>
                  {videoInfo.duration && (
                    <div className="text-gray-400 text-xs mt-0.5">{formatDuration(videoInfo.duration)}</div>
                  )}
                </div>
              </div>
            )}

            {/* Row 2: Dataset + Format + Add button */}
            <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-end">
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

              {/* Resolution selector — always visible once info is fetched, otherwise a free-text input */}
              <div className="w-full sm:w-56">
                <label className="block text-xs mb-1 mt-2 text-gray-300">Resolution / Format</label>
                {videoInfo && videoInfo.resolutions.length > 0 ? (
                  <select
                    value={newFormat}
                    onChange={e => setNewFormat(e.target.value)}
                    className="w-full text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                  >
                    {videoInfo.resolutions.map(r => (
                      <option key={r.format} value={r.format}>
                        {r.label}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={newFormat}
                    onChange={e => setNewFormat(e.target.value)}
                    placeholder="best (default)"
                    className="w-full text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                  />
                )}
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
            </div>

            {/* Advanced section: cookies */}
            <div className="mt-3">
              <button
                type="button"
                onClick={() => setShowAdvanced(v => !v)}
                className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                {showAdvanced ? <FaChevronUp /> : <FaChevronDown />}
                Advanced
              </button>

              {showAdvanced && (
                <div className="mt-2">
                  <TextInput
                    label="Cookies file path (optional)"
                    value={newCookiesFile}
                    onChange={setNewCookiesFile}
                    placeholder="/path/to/cookies.txt"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Netscape-format cookies file exported from your browser. Helps with authenticated sites and
                    avoids throttled downloads.
                  </p>
                </div>
              )}
            </div>

            {formError && <p className="mt-2 text-sm text-red-400">{formError}</p>}
          </form>
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
