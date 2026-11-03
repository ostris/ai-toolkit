'use client';
import React, { useEffect, useMemo, useState } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Loader2, AlertCircle, Search } from 'lucide-react';
import { Modal } from './Modal';
import { getFilename } from '@/utils/basic';
import { apiClient } from '@/utils/api';

export interface SafetensorsInfoModalState {
  filePath: string;
}

interface TensorInfo {
  name: string;
  dtype: string;
  shape: number[];
  numel: number;
  bytes: number;
}

interface SafetensorsInfo {
  filePath: string;
  fileSize: number;
  headerSize: number;
  metadata: Record<string, string>;
  tensors: TensorInfo[];
  tensorCount: number;
  totalParams: number;
  totalBytes: number;
  dtypeCounts: Record<string, { count: number; numel: number; bytes: number }>;
}

export const safetensorsInfoModalState = createGlobalState<SafetensorsInfoModalState | null>(null);

export const openSafetensorsInfoModal = (filePath: string) => {
  safetensorsInfoModalState.set({ filePath });
};

const fmtBytes = (size: number) => {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  return `${(size / (1024 * 1024 * 1024)).toFixed(2)} GB`;
};

const fmtCount = (n: number) => {
  if (n < 1000) return `${n}`;
  if (n < 1e6) return `${(n / 1e3).toFixed(1)}K`;
  if (n < 1e9) return `${(n / 1e6).toFixed(1)}M`;
  return `${(n / 1e9).toFixed(2)}B`;
};

// metadata values are strings; many are JSON blobs worth pretty-printing
const fmtMetaValue = (value: string) => {
  const trimmed = value.trim();
  if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
    try {
      return JSON.stringify(JSON.parse(trimmed), null, 2);
    } catch {
      /* not json */
    }
  }
  return value;
};

const MAX_ROWS = 2000;

const SafetensorsInfoModal: React.FC = () => {
  const [modalInfo, setModalInfo] = safetensorsInfoModalState.use();
  const isOpen = modalInfo !== null;
  const [info, setInfo] = useState<SafetensorsInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [filter, setFilter] = useState('');
  const [tab, setTab] = useState<'tensors' | 'metadata'>('tensors');

  useEffect(() => {
    if (!modalInfo) {
      setInfo(null);
      setError(null);
      setIsLoading(false);
      setFilter('');
      setTab('tensors');
      return;
    }
    let cancelled = false;
    setIsLoading(true);
    setInfo(null);
    setError(null);
    apiClient
      .post('/api/files/info', { filePath: modalInfo.filePath })
      .then(res => {
        if (cancelled) return;
        setInfo(res.data as SafetensorsInfo);
      })
      .catch(err => {
        if (cancelled) return;
        setError(err?.response?.data || err?.message || 'Failed to read file');
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [modalInfo]);

  const onClose = () => setModalInfo(null);

  const filteredTensors = useMemo(() => {
    if (!info) return [];
    const q = filter.trim().toLowerCase();
    if (!q) return info.tensors;
    return info.tensors.filter(t => t.name.toLowerCase().includes(q) || t.dtype.toLowerCase().includes(q));
  }, [info, filter]);

  const metadataEntries = info ? Object.entries(info.metadata) : [];
  const title = modalInfo ? getFilename(modalInfo.filePath) : 'Safetensors Info';

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={title} size="xl">
      {isLoading && (
        <div className="flex flex-col items-center justify-center py-16 space-y-3 text-gray-300">
          <Loader2 className="w-8 h-8 text-purple-400 animate-spin" />
          <span className="text-sm">Gathering info...</span>
        </div>
      )}

      {!isLoading && error && (
        <div className="flex items-center justify-center py-10 text-rose-400 space-x-2">
          <AlertCircle className="w-5 h-5" />
          <span className="text-sm">{String(error)}</span>
        </div>
      )}

      {!isLoading && info && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="bg-gray-800 rounded-lg px-3 py-2">
              <div className="text-gray-500">File size</div>
              <div className="text-gray-100 font-mono">{fmtBytes(info.fileSize)}</div>
            </div>
            <div className="bg-gray-800 rounded-lg px-3 py-2">
              <div className="text-gray-500">Tensors</div>
              <div className="text-gray-100 font-mono">{info.tensorCount.toLocaleString()}</div>
            </div>
            <div className="bg-gray-800 rounded-lg px-3 py-2">
              <div className="text-gray-500">Parameters</div>
              <div className="text-gray-100 font-mono" title={info.totalParams.toLocaleString()}>
                {fmtCount(info.totalParams)}
              </div>
            </div>
            <div className="bg-gray-800 rounded-lg px-3 py-2">
              <div className="text-gray-500">Header</div>
              <div className="text-gray-100 font-mono">{fmtBytes(info.headerSize)}</div>
            </div>
          </div>

          <div className="flex flex-wrap gap-1.5 text-xs">
            {Object.entries(info.dtypeCounts)
              .sort((a, b) => b[1].bytes - a[1].bytes)
              .map(([dtype, d]) => (
                <span
                  key={dtype}
                  className="px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-300 font-mono"
                  title={`${d.count} tensors · ${d.numel.toLocaleString()} params`}
                >
                  {dtype} · {d.count} · {fmtBytes(d.bytes)}
                </span>
              ))}
          </div>

          <div className="flex items-center justify-between border-b border-gray-800">
            <div className="flex space-x-1">
              <button
                type="button"
                onClick={() => setTab('tensors')}
                className={`px-3 py-1.5 text-sm border-b-2 ${
                  tab === 'tensors'
                    ? 'border-purple-500 text-gray-100'
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
              >
                Tensors
              </button>
              <button
                type="button"
                onClick={() => setTab('metadata')}
                className={`px-3 py-1.5 text-sm border-b-2 ${
                  tab === 'metadata'
                    ? 'border-purple-500 text-gray-100'
                    : 'border-transparent text-gray-400 hover:text-gray-200'
                }`}
              >
                Metadata
                <span className="ml-1.5 px-1.5 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300">
                  {metadataEntries.length}
                </span>
              </button>
            </div>
            {tab === 'tensors' && (
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-gray-500 absolute left-2 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={filter}
                  onChange={e => setFilter(e.target.value)}
                  placeholder="Filter tensors"
                  className="text-xs pl-7 pr-2 py-1 w-56 bg-gray-950 dark:bg-gray-800 border border-gray-700 rounded-md text-gray-100 focus:ring-1 focus:ring-gray-600 focus:outline-none"
                />
              </div>
            )}
          </div>

          {tab === 'tensors' && (
            <div className="overflow-auto max-h-[60vh] rounded-md border border-gray-800">
              <table className="w-full text-xs font-mono">
                <thead className="sticky top-0 bg-gray-800 text-gray-400">
                  <tr>
                    <th className="text-left px-2 py-1.5 font-normal">Name</th>
                    <th className="text-left px-2 py-1.5 font-normal">Shape</th>
                    <th className="text-left px-2 py-1.5 font-normal">Dtype</th>
                    <th className="text-right px-2 py-1.5 font-normal">Params</th>
                    <th className="text-right px-2 py-1.5 font-normal">Size</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTensors.slice(0, MAX_ROWS).map(t => (
                    <tr key={t.name} className="border-t border-gray-800/60 hover:bg-gray-800/50">
                      <td className="px-2 py-1 text-gray-200 break-all">{t.name}</td>
                      <td className="px-2 py-1 text-gray-300 whitespace-nowrap">[{t.shape.join(', ')}]</td>
                      <td className="px-2 py-1 text-purple-300 whitespace-nowrap">{t.dtype}</td>
                      <td className="px-2 py-1 text-gray-400 text-right whitespace-nowrap">
                        {t.numel.toLocaleString()}
                      </td>
                      <td className="px-2 py-1 text-gray-400 text-right whitespace-nowrap">{fmtBytes(t.bytes)}</td>
                    </tr>
                  ))}
                  {filteredTensors.length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-2 py-4 text-center text-gray-500">
                        No tensors match
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
              {filteredTensors.length > MAX_ROWS && (
                <div className="px-2 py-1.5 text-xs text-gray-500 bg-gray-800 border-t border-gray-700">
                  Showing {MAX_ROWS.toLocaleString()} of {filteredTensors.length.toLocaleString()} — narrow with the
                  filter
                </div>
              )}
            </div>
          )}

          {tab === 'metadata' && (
            <div className="overflow-auto max-h-[60vh] rounded-md border border-gray-800">
              {metadataEntries.length === 0 ? (
                <div className="px-2 py-4 text-center text-gray-500 text-xs">No metadata</div>
              ) : (
                <div className="divide-y divide-gray-800/60">
                  {metadataEntries.map(([key, value]) => (
                    <div key={key} className="px-2 py-1.5">
                      <div className="text-xs text-purple-300 font-mono">{key}</div>
                      <pre className="text-xs text-gray-200 font-mono whitespace-pre-wrap break-all mt-0.5">
                        {fmtMetaValue(String(value))}
                      </pre>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Modal>
  );
};

export default SafetensorsInfoModal;
