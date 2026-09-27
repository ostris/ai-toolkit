'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Modal } from '@/components/Modal';
import { apiClient } from '@/utils/api';

// ─── Types (mirror Python dataclasses) ───────────────────────────────────────

interface SourceItem {
  id: string;
  name: string;
  picture_count: number;
  thumbnail_id: string;
  thumbnail_type: string;
}

interface SourceGroup {
  id: string;
  label: string;
  items: SourceItem[];
}

interface ImportFieldOption {
  value: string | number;
  label: string;
}

interface ImportField {
  id: string;
  label: string;
  field_type: 'select' | 'text' | 'checkbox';
  options: ImportFieldOption[];
  default: string | number | boolean | null;
  required: boolean;
}

interface BrowseData {
  groups: SourceGroup[];
  import_fields: ImportField[];
}

// ─── Thumbnail image ──────────────────────────────────────────────────────────

function ThumbnailImage({
  sourceId,
  thumbnailId,
  thumbnailType,
  alt,
}: {
  sourceId: string;
  thumbnailId: string;
  thumbnailType: string;
  alt: string;
}) {
  const [errored, setErrored] = useState(false);
  const src = `/api/datasets/remote/${sourceId}/thumbnail/${encodeURIComponent(thumbnailId)}?type=${thumbnailType}`;
  if (!thumbnailId || errored) {
    return (
      <div className="w-16 h-16 rounded bg-gray-700 flex items-center justify-center text-gray-500 text-xs">?</div>
    );
  }
  return (
    <img
      src={src}
      alt={alt}
      className="w-16 h-16 rounded object-cover flex-shrink-0"
      onError={() => setErrored(true)}
    />
  );
}

// ─── Progress bar ─────────────────────────────────────────────────────────────

function ProgressBar({ done, total }: { done: number; total: number }) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>
          {done} / {total} images
        </span>
        <span>{pct}%</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
        <div
          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ─── Import form ──────────────────────────────────────────────────────────────

type ImportPhase = 'idle' | 'connecting' | 'downloading' | 'complete' | 'error';

interface ImportFormProps {
  sourcePluginId: string;
  groupId: string;
  itemId: string;
  itemName: string;
  importFields: ImportField[];
  onSuccess: (datasetName: string) => void;
  onCancel: () => void;
}

function ImportForm({
  sourcePluginId,
  groupId,
  itemId,
  itemName,
  importFields,
  onSuccess,
  onCancel,
}: ImportFormProps) {
  const [triggerWord, setTriggerWord] = useState('');
  const [datasetName, setDatasetName] = useState(itemName);
  const [overwrite, setOverwrite] = useState(false);
  const [phase, setPhase] = useState<ImportPhase>('idle');
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [downloaded, setDownloaded] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Initialise source-specific field values from their declared defaults
  const [fieldValues, setFieldValues] = useState<Record<string, string | number | boolean>>(() => {
    const init: Record<string, string | number | boolean> = {};
    for (const f of importFields) {
      init[f.id] =
        f.default !== null && f.default !== undefined
          ? f.default
          : f.field_type === 'checkbox'
            ? false
            : (f.options[0]?.value ?? '');
    }
    return init;
  });

  const handleImport = async () => {
    setPhase('connecting');
    setErrorMsg(null);
    setProgress({ done: 0, total: 0 });

    try {
      const token =
        typeof window !== 'undefined' ? localStorage.getItem('AI_TOOLKIT_AUTH') : null;
      const res = await fetch(`/api/datasets/remote/${sourcePluginId}/import`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          source_type: groupId,
          source_id: itemId,
          trigger_word: triggerWord || undefined,
          dataset_name: datasetName || undefined,
          overwrite,
          ...fieldValues,
        }),
      });

      if (!res.ok || !res.body) {
        const errBody = await res.json().catch(() => ({ error: 'Import failed' }));
        setPhase('error');
        setErrorMsg(errBody.error || 'Import failed');
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const eventLine = part.split('\n').find(l => l.startsWith('event:'));
          const dataLine = part.split('\n').find(l => l.startsWith('data:'));
          if (!dataLine) continue;

          const eventType = eventLine ? eventLine.slice(7).trim() : 'message';
          const data = JSON.parse(dataLine.slice(5).trim());

          if (eventType === 'total') {
            setPhase('downloading');
            setProgress({ done: 0, total: data.count });
          } else if (eventType === 'progress') {
            setPhase('downloading');
            setProgress({ done: data.done, total: data.total });
          } else if (eventType === 'complete') {
            setDownloaded(data.downloaded);
            setPhase('complete');
            onSuccess(datasetName || itemName);
          } else if (eventType === 'error') {
            setPhase('error');
            setErrorMsg(data.message || 'Download failed');
          }
        }
      }
    } catch (err: unknown) {
      setPhase('error');
      setErrorMsg(err instanceof Error ? err.message : 'Unexpected error');
    }
  };

  const inputClass =
    'w-full rounded bg-gray-700 border border-gray-600 px-3 py-2 text-gray-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500';
  const labelClass = 'block text-sm text-gray-400 mb-1';
  const isRunning = phase === 'connecting' || phase === 'downloading';

  if (phase === 'complete') {
    return (
      <div className="p-4 border border-green-800 rounded-lg bg-green-950/40 space-y-3">
        <div className="text-sm text-green-300 font-medium">
          ✓ Downloaded {downloaded} image{downloaded !== 1 ? 's' : ''} to &ldquo;
          {datasetName || itemName}&rdquo;
        </div>
        <ProgressBar done={downloaded} total={downloaded} />
        <button
          className="w-full rounded-md bg-gray-700 px-4 py-2 text-sm text-gray-200 hover:bg-gray-600"
          onClick={onCancel}
        >
          Close
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4 p-4 border border-gray-700 rounded-lg bg-gray-900">
      <div className="text-sm font-medium text-gray-200">
        Downloading: <span className="text-blue-400">{itemName}</span>
      </div>

      {!isRunning && (
        <>
          <div>
            <label className={labelClass}>Dataset Name</label>
            <input
              type="text"
              className={inputClass}
              value={datasetName}
              onChange={e => setDatasetName(e.target.value)}
              placeholder={itemName}
            />
            <p className="text-xs text-gray-500 mt-1">
              Folder name created inside your datasets directory.
            </p>
          </div>

          <div>
            <label className={labelClass}>Trigger Word (optional)</label>
            <input
              type="text"
              className={inputClass}
              value={triggerWord}
              onChange={e => setTriggerWord(e.target.value)}
              placeholder="e.g. myperson"
            />
            <p className="text-xs text-gray-500 mt-1">
              Prepended to every caption for LoRA training.
            </p>
          </div>

          {/* Source-specific fields from plugin's get_import_fields() */}
          {importFields.map(f => (
            <div key={f.id}>
              <label className={labelClass}>{f.label}</label>
              {f.field_type === 'select' && (
                <select
                  className={inputClass}
                  value={String(fieldValues[f.id] ?? f.default ?? '')}
                  onChange={e =>
                    setFieldValues(v => ({
                      ...v,
                      [f.id]: isNaN(Number(e.target.value)) ? e.target.value : Number(e.target.value),
                    }))
                  }
                >
                  {f.options.map(opt => (
                    <option key={String(opt.value)} value={String(opt.value)}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              )}
              {f.field_type === 'text' && (
                <input
                  type="text"
                  className={inputClass}
                  value={String(fieldValues[f.id] ?? '')}
                  onChange={e => setFieldValues(v => ({ ...v, [f.id]: e.target.value }))}
                />
              )}
              {f.field_type === 'checkbox' && (
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={Boolean(fieldValues[f.id])}
                    onChange={e => setFieldValues(v => ({ ...v, [f.id]: e.target.checked }))}
                    className="rounded"
                  />
                  <span className="text-sm text-gray-300">Enabled</span>
                </label>
              )}
            </div>
          ))}

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={overwrite}
              onChange={e => setOverwrite(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm text-gray-300">Overwrite existing files</span>
          </label>
        </>
      )}

      {isRunning && (
        <div className="space-y-3 py-2">
          <div className="text-sm text-gray-300">
            {phase === 'connecting'
              ? 'Connecting…'
              : `Downloading to "${datasetName || itemName}"…`}
          </div>
          {progress.total > 0 ? (
            <ProgressBar done={progress.done} total={progress.total} />
          ) : (
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <div className="bg-blue-500 h-2 rounded-full animate-pulse w-1/3" />
            </div>
          )}
        </div>
      )}

      {errorMsg && (
        <div className="text-sm text-red-400 bg-red-950 border border-red-800 rounded px-3 py-2">
          {errorMsg}
        </div>
      )}

      <div className="flex justify-end gap-3 pt-1">
        <button
          type="button"
          className="rounded-md bg-gray-700 px-4 py-2 text-sm text-gray-200 hover:bg-gray-600 disabled:opacity-40"
          onClick={onCancel}
          disabled={isRunning}
        >
          Cancel
        </button>
        <button
          type="button"
          className="rounded-md bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
          onClick={handleImport}
          disabled={isRunning}
        >
          {phase === 'connecting'
            ? 'Connecting…'
            : phase === 'downloading'
              ? 'Downloading…'
              : 'Download to Dataset'}
        </button>
      </div>
    </div>
  );
}

// ─── Main modal ───────────────────────────────────────────────────────────────

interface RemoteSourceBrowseModalProps {
  /** Plugin id, e.g. "pixlstash" */
  sourceId: string;
  /** Human-readable name shown in the modal title, e.g. "PixlStash" */
  sourceName: string;
  isOpen: boolean;
  onClose: () => void;
  onImportStarted?: () => void;
}

export default function RemoteSourceBrowseModal({
  sourceId,
  sourceName,
  isOpen,
  onClose,
  onImportStarted,
}: RemoteSourceBrowseModalProps) {
  const router = useRouter();
  const [browseData, setBrowseData] = useState<BrowseData | null>(null);
  const [browseError, setBrowseError] = useState<string | null>(null);
  const [browseLoading, setBrowseLoading] = useState(false);
  const [activeGroup, setActiveGroup] = useState<string>('');
  const [selectedItem, setSelectedItem] = useState<{
    id: string;
    name: string;
    groupId: string;
  } | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    if (!isOpen) return;
    setSelectedItem(null);

    let aborted = false;
    setBrowseLoading(true);
    setBrowseError(null);

    apiClient
      .get(`/api/datasets/remote/${sourceId}/browse`)
      .then(res => {
        if (aborted) return;
        const data: BrowseData = res.data;
        setBrowseData(data);
        if (data.groups?.length > 0) setActiveGroup(data.groups[0].id);
      })
      .catch(err => {
        if (aborted) return;
        setBrowseError(
          (err as any)?.response?.data?.error ||
            (err instanceof Error ? err.message : 'Failed to connect'),
        );
      })
      .finally(() => {
        if (!aborted) setBrowseLoading(false);
      });

    return () => {
      aborted = true;
    };
  }, [isOpen, sourceId, retryCount]);

  const handleImportSuccess = (datasetName: string) => {
    onImportStarted?.();
    onClose();
    router.push(`/datasets/${encodeURIComponent(datasetName)}`);
  };

  const tabClass = (groupId: string) =>
    `px-4 py-2 text-sm font-medium rounded-t border-b-2 transition-colors ${
      activeGroup === groupId
        ? 'border-blue-500 text-blue-400'
        : 'border-transparent text-gray-400 hover:text-gray-200'
    }`;

  const currentGroup = browseData?.groups.find(g => g.id === activeGroup);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`Browse ${sourceName}`} size="lg">
      <div className="flex flex-col" style={{ minHeight: '500px' }}>
        {/* Group tabs */}
        {browseData && browseData.groups.length > 1 && (
          <div className="flex border-b border-gray-700 mb-3">
            {browseData.groups.map(g => (
              <button
                key={g.id}
                className={tabClass(g.id)}
                onClick={() => {
                  setActiveGroup(g.id);
                  setSelectedItem(null);
                }}
              >
                {g.label}
              </button>
            ))}
          </div>
        )}

        {/* Browse list */}
        {!selectedItem && (
          <>
            {browseLoading && (
              <div className="py-12 text-center text-gray-400 text-sm">
                Loading {sourceName} data…
              </div>
            )}
            {browseError && (
              <div className="py-8 text-center">
                <div className="text-red-400 text-sm mb-3">{browseError}</div>
                <button
                  className="text-xs text-blue-400 hover:text-blue-300 underline"
                  onClick={() => setRetryCount(c => c + 1)}
                >
                  Retry
                </button>
              </div>
            )}
            {currentGroup && (
              <div className="flex-1 overflow-y-auto max-h-96">
                {currentGroup.items.length === 0 ? (
                  <div className="py-12 text-center text-gray-500 text-sm">
                    No {currentGroup.label.toLowerCase()} found.
                  </div>
                ) : (
                  <ul className="divide-y divide-gray-700">
                    {currentGroup.items.map(item => (
                      <li key={item.id}>
                        <button
                          className="flex items-center gap-4 w-full px-4 py-3 text-left hover:bg-gray-700 transition-colors"
                          onClick={() =>
                            setSelectedItem({
                              id: item.id,
                              name: item.name,
                              groupId: currentGroup.id,
                            })
                          }
                        >
                          <ThumbnailImage
                            sourceId={sourceId}
                            thumbnailId={item.thumbnail_id}
                            thumbnailType={item.thumbnail_type}
                            alt={item.name}
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-gray-100 truncate">
                                {item.name}
                              </span>
                              {item.picture_count > 0 && (
                                <span className="text-xs text-gray-500 flex-shrink-0">
                                  {item.picture_count} images
                                </span>
                              )}
                            </div>
                          </div>
                          <span className="text-gray-500 text-sm flex-shrink-0">▶</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </>
        )}

        {/* Import form */}
        {selectedItem && (
          <div>
            <button
              className="text-xs text-blue-400 hover:text-blue-300 mb-3 flex items-center gap-1"
              onClick={() => setSelectedItem(null)}
            >
              ← Back to list
            </button>
            <ImportForm
              sourcePluginId={sourceId}
              groupId={selectedItem.groupId}
              itemId={selectedItem.id}
              itemName={selectedItem.name}
              importFields={browseData?.import_fields ?? []}
              onSuccess={handleImportSuccess}
              onCancel={() => setSelectedItem(null)}
            />
          </div>
        )}
      </div>
    </Modal>
  );
}
