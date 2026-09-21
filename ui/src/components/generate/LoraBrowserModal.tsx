'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, ChevronRight, Cloud, FolderOpen, Loader2, Search, Upload, X } from 'lucide-react';
import { Modal } from '@/components/Modal';
import { apiClient } from '@/utils/api';
import { CloudLora } from '@/types';

export interface LoraPick {
  path: string;
  name: string;
}

interface LoraFile {
  name: string;
  path: string;
  size: number;
  mtime: number;
  relpath?: string;
}

interface JobEntry {
  id: string;
  name: string;
  status: string;
  updated_at: string;
  files: LoraFile[];
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onPick: (lora: LoraPick) => void;
  // LoRAs published for the selected model. Picked by their hub reference; the
  // backend finds or downloads the file when the job runs.
  cloudLoras?: CloudLora[];
}

type Tab = 'cloud' | 'jobs' | 'models' | 'upload';

const fmtSize = (n: number) => (n > 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${(n / 1e6).toFixed(0)} MB`);
const fmtDate = (ms: number) => new Date(ms).toLocaleString();

const CHUNK_SIZE = 16 * 1024 * 1024;
const CHUNK_RETRIES = 3;

interface UploadState {
  file: File;
  sent: number;
  startedAt: number;
  status: 'uploading' | 'done' | 'error' | 'cancelled';
  error?: string;
  result?: { path: string; name: string };
}

const isAbort = (e: any) => e?.code === 'ERR_CANCELED' || e?.name === 'CanceledError' || e?.name === 'AbortError';

export default function LoraBrowserModal({ isOpen, onClose, onPick, cloudLoras }: Props) {
  const hasCloud = !!cloudLoras?.length;
  const [tab, setTab] = useState<Tab>(hasCloud ? 'cloud' : 'jobs');
  const [loading, setLoading] = useState(false);
  const [jobs, setJobs] = useState<JobEntry[]>([]);
  const [models, setModels] = useState<LoraFile[]>([]);
  const [lorasRoot, setLorasRoot] = useState('');
  const [filter, setFilter] = useState('');
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [upload, setUpload] = useState<UploadState | null>(null);
  const [overwrite, setOverwrite] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isUploading = upload?.status === 'uploading';

  const refresh = useCallback(() => {
    setLoading(true);
    return apiClient
      .get('/api/loras')
      .then(r => {
        setJobs(r.data.jobs || []);
        setModels(r.data.models || []);
        setLorasRoot(r.data.lorasRoot || '');
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    setTab(hasCloud ? 'cloud' : 'jobs');
    setUpload(null);
    setOverwrite(false);
    refresh();
  }, [isOpen]);

  // Modal only closes via onClose; while a transfer runs, every close path is a no-op.
  const guardedClose = useCallback(() => {
    if (isUploading) return;
    onClose();
  }, [isUploading, onClose]);

  useEffect(() => {
    if (!isUploading) return;
    const warn = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [isUploading]);

  const startUpload = async (file: File) => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setUpload({ file, sent: 0, startedAt: Date.now(), status: 'uploading' });
    const params = (extra: Record<string, string | number>) => ({ overwrite: overwrite ? 1 : 0, ...extra });
    let uploadId = '';
    try {
      const start = await apiClient.post('/api/loras/upload', null, {
        params: params({ action: 'start', fileName: file.name, size: file.size }),
        signal: ctrl.signal,
      });
      uploadId = start.data.uploadId;
      for (let offset = 0; offset < file.size; offset += CHUNK_SIZE) {
        const chunk = file.slice(offset, Math.min(offset + CHUNK_SIZE, file.size));
        let attempt = 0;
        for (;;) {
          try {
            await apiClient.post('/api/loras/upload', chunk, {
              params: { action: 'chunk', uploadId, offset },
              headers: { 'Content-Type': 'application/octet-stream' },
              signal: ctrl.signal,
              onUploadProgress: ev => setUpload(u => (u ? { ...u, sent: offset + (ev.loaded || 0) } : u)),
            });
            break;
          } catch (e: any) {
            // Only transport failures are retried; the server rejects bad requests with 4xx.
            if (isAbort(e) || e?.response || ++attempt >= CHUNK_RETRIES) throw e;
            await new Promise(r => setTimeout(r, 1000 * attempt));
          }
        }
        setUpload(u => (u ? { ...u, sent: offset + chunk.size } : u));
      }
      const fin = await apiClient.post('/api/loras/upload', null, {
        params: params({ action: 'finish', uploadId, fileName: file.name, size: file.size }),
        signal: ctrl.signal,
      });
      setUpload(u => (u ? { ...u, status: 'done', sent: file.size, result: fin.data } : u));
      refresh();
    } catch (e: any) {
      if (uploadId)
        apiClient.post('/api/loras/upload', null, { params: { action: 'cancel', uploadId } }).catch(() => {});
      if (isAbort(e)) {
        setUpload(u => (u ? { ...u, status: 'cancelled' } : u));
      } else {
        const msg = e?.response?.data?.error || e?.message || 'Upload failed';
        setUpload(u => (u ? { ...u, status: 'error', error: msg } : u));
      }
    } finally {
      abortRef.current = null;
    }
  };

  const cancelUpload = () => abortRef.current?.abort();

  const onFileChosen = (files: FileList | null) => {
    const f = files?.[0];
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (!f) return;
    if (!f.name.toLowerCase().endsWith('.safetensors')) {
      setUpload({
        file: f,
        sent: 0,
        startedAt: Date.now(),
        status: 'error',
        error: 'Only .safetensors files can be uploaded',
      });
      return;
    }
    startUpload(f);
  };

  const q = filter.trim().toLowerCase();
  const filteredJobs = useMemo(
    () =>
      q
        ? jobs.filter(j => j.name.toLowerCase().includes(q) || j.files.some(f => f.name.toLowerCase().includes(q)))
        : jobs,
    [jobs, q],
  );
  const filteredModels = useMemo(
    () => (q ? models.filter(m => (m.relpath || m.name).toLowerCase().includes(q)) : models),
    [models, q],
  );
  const filteredCloud = useMemo(() => {
    const list = cloudLoras || [];
    return q ? list.filter(c => c.name.toLowerCase().includes(q) || c.path.toLowerCase().includes(q)) : list;
  }, [cloudLoras, q]);

  const pickCloud = (c: CloudLora) => {
    onPick({ path: c.path, name: c.name });
    onClose();
  };

  const pick = (f: LoraFile, jobName?: string) => {
    onPick({
      path: f.path,
      name: jobName
        ? `${jobName} / ${f.name.replace(/\.safetensors$/, '')}`
        : (f.relpath || f.name).replace(/\.safetensors$/, ''),
    });
    onClose();
  };

  const tabClass = (t: Tab) =>
    `px-3 py-1.5 text-sm border-b-2 disabled:opacity-50 ${tab === t ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-400 hover:text-gray-200'}`;

  const pct = upload ? (upload.file.size ? Math.min(100, (upload.sent / upload.file.size) * 100) : 100) : 0;
  const elapsed = upload ? (Date.now() - upload.startedAt) / 1000 : 0;
  const rate = upload && elapsed > 0 ? upload.sent / elapsed : 0;
  const eta = rate > 0 && upload ? (upload.file.size - upload.sent) / rate : 0;

  return (
    <Modal
      isOpen={isOpen}
      onClose={guardedClose}
      title="Add LoRA"
      size="lg"
      showCloseButton={!isUploading}
      closeOnOverlayClick={!isUploading}
    >
      <div className="text-gray-200">
        <div className="flex items-center gap-2 border-b border-gray-800 mb-3">
          {hasCloud && (
            <button type="button" className={tabClass('cloud')} onClick={() => setTab('cloud')} disabled={isUploading}>
              For this model
            </button>
          )}
          <button type="button" className={tabClass('jobs')} onClick={() => setTab('jobs')} disabled={isUploading}>
            Training jobs
          </button>
          <button type="button" className={tabClass('models')} onClick={() => setTab('models')} disabled={isUploading}>
            Models folder
          </button>
          <button type="button" className={tabClass('upload')} onClick={() => setTab('upload')} disabled={isUploading}>
            Upload
          </button>
          <div className="flex-1" />
          <div className="relative mb-1">
            <Search className="w-3.5 h-3.5 absolute left-2 top-2 text-gray-500" />
            <input
              value={filter}
              onChange={e => setFilter(e.target.value)}
              placeholder="filter"
              className="pl-7 pr-2 py-1 text-xs bg-gray-950 border border-gray-700 rounded text-gray-100 w-48"
            />
          </div>
        </div>
        <div className="max-h-[60vh] overflow-y-auto pr-1">
          {loading && (
            <div className="text-gray-500 text-sm flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading…
            </div>
          )}
          {tab === 'cloud' && (
            <div>
              <div className="text-[11px] text-gray-500 mb-2 flex items-center gap-1">
                <Cloud className="w-3.5 h-3.5" /> Downloaded to the loras folder on first use
              </div>
              {filteredCloud.length === 0 && (
                <div className="text-xs text-gray-500">No published LoRAs for this model.</div>
              )}
              <div className="space-y-0.5">
                {filteredCloud.map(c => (
                  <button
                    type="button"
                    key={c.path}
                    onClick={() => pickCloud(c)}
                    className="w-full px-2 py-1 rounded text-xs text-left hover:bg-blue-900/40"
                    title={c.path}
                  >
                    <div className="truncate text-gray-200">{c.name}</div>
                    <div className="truncate font-mono text-[11px] text-gray-500">{c.path}</div>
                    {c.description && <div className="truncate text-[11px] text-gray-500">{c.description}</div>}
                  </button>
                ))}
              </div>
            </div>
          )}
          {!loading && tab === 'jobs' && (
            <div className="space-y-1">
              {filteredJobs.length === 0 && (
                <div className="text-xs text-gray-500">No training jobs with saved LoRAs.</div>
              )}
              {filteredJobs.map(j => {
                const isOpen = open.has(j.id) || !!q;
                return (
                  <div key={j.id} className="rounded-md bg-gray-900 border border-gray-800">
                    <button
                      type="button"
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm text-left hover:bg-gray-800/60"
                      onClick={() =>
                        setOpen(prev => {
                          const n = new Set(prev);
                          if (n.has(j.id)) n.delete(j.id);
                          else n.add(j.id);
                          return n;
                        })
                      }
                    >
                      {isOpen ? (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-gray-500" />
                      )}
                      <span className="truncate flex-1">{j.name}</span>
                      <span className="text-[11px] text-gray-500">
                        {j.files.length} file{j.files.length === 1 ? '' : 's'}
                      </span>
                    </button>
                    {isOpen && (
                      <div className="px-2 pb-2 space-y-0.5">
                        {j.files
                          .filter(f => !q || f.name.toLowerCase().includes(q) || j.name.toLowerCase().includes(q))
                          .map(f => (
                            <button
                              type="button"
                              key={f.path}
                              onClick={() => pick(f, j.name)}
                              className="w-full flex items-center gap-2 px-2 py-1 rounded text-xs text-left hover:bg-blue-900/40"
                              title={f.path}
                            >
                              <span className="truncate flex-1 font-mono">{f.name}</span>
                              <span className="text-gray-500 shrink-0">{fmtSize(f.size)}</span>
                              <span className="text-gray-600 shrink-0 hidden sm:inline">{fmtDate(f.mtime)}</span>
                            </button>
                          ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
          {tab === 'upload' && (
            <div>
              <div className="text-[11px] text-gray-500 mb-2 flex items-center gap-1">
                <FolderOpen className="w-3.5 h-3.5" /> Uploads to {lorasRoot || 'MODELS_PATH/loras'}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".safetensors"
                className="hidden"
                onChange={e => onFileChosen(e.target.files)}
              />
              {!upload || upload.status === 'cancelled' || upload.status === 'error' ? (
                <div>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={e => e.preventDefault()}
                    onDrop={e => {
                      e.preventDefault();
                      onFileChosen(e.dataTransfer.files);
                    }}
                    className="w-full flex flex-col items-center justify-center gap-2 py-8 rounded-md border border-dashed border-gray-700 hover:border-blue-500 hover:bg-gray-900/60 text-sm text-gray-400"
                  >
                    <Upload className="w-6 h-6" />
                    <span>Click to choose a .safetensors file, or drop one here</span>
                    <span className="text-[11px] text-gray-500">
                      Large files are sent in {fmtSize(CHUNK_SIZE)} chunks
                    </span>
                  </button>
                  <label className="mt-2 flex items-center gap-2 text-xs text-gray-400">
                    <input type="checkbox" checked={overwrite} onChange={e => setOverwrite(e.target.checked)} />
                    Overwrite if a file with the same name exists
                  </label>
                  {upload?.status === 'error' && (
                    <div className="mt-2 text-xs text-red-400">
                      {upload.file.name}: {upload.error}
                    </div>
                  )}
                  {upload?.status === 'cancelled' && (
                    <div className="mt-2 text-xs text-gray-500">Upload of {upload.file.name} cancelled.</div>
                  )}
                </div>
              ) : (
                <div className="rounded-md bg-gray-900 border border-gray-800 p-3">
                  <div className="flex items-center gap-2 text-sm">
                    {upload.status === 'uploading' && <Loader2 className="w-4 h-4 animate-spin text-blue-400" />}
                    <span className="truncate flex-1 font-mono">{upload.file.name}</span>
                    <span className="text-gray-500 shrink-0">{fmtSize(upload.file.size)}</span>
                  </div>
                  <div className="mt-2 h-2 w-full rounded bg-gray-800 overflow-hidden">
                    <div
                      className={`h-full transition-[width] ${upload.status === 'done' ? 'bg-green-500' : 'bg-blue-500'}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-gray-500">
                    <span>
                      {fmtSize(upload.sent)} / {fmtSize(upload.file.size)} ({pct.toFixed(1)}%)
                    </span>
                    {upload.status === 'uploading' && (
                      <span>
                        {fmtSize(rate)}/s{eta > 0 ? ` · ~${Math.ceil(eta)}s left` : ''}
                      </span>
                    )}
                  </div>
                  {upload.status === 'uploading' && (
                    <div className="mt-3 flex items-center justify-between">
                      <span className="text-xs text-amber-400">
                        Uploading… this window stays open until the transfer finishes.
                      </span>
                      <button
                        type="button"
                        onClick={cancelUpload}
                        className="flex items-center gap-1 px-3 py-1 rounded text-xs bg-red-900/60 hover:bg-red-800 text-red-100"
                      >
                        <X className="w-3.5 h-3.5" /> Cancel
                      </button>
                    </div>
                  )}
                  {upload.status === 'done' && upload.result && (
                    <div className="mt-3 flex items-center justify-between">
                      <span className="text-xs text-green-400">Upload complete.</span>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => setUpload(null)}
                          className="px-3 py-1 rounded text-xs bg-gray-800 hover:bg-gray-700"
                        >
                          Upload another
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            pick({
                              name: upload.result!.name,
                              path: upload.result!.path,
                              size: upload.file.size,
                              mtime: Date.now(),
                            })
                          }
                          className="px-3 py-1 rounded text-xs bg-blue-700 hover:bg-blue-600 text-white"
                        >
                          Use this LoRA
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
          {!loading && tab === 'models' && (
            <div>
              <div className="text-[11px] text-gray-500 mb-2 flex items-center gap-1">
                <FolderOpen className="w-3.5 h-3.5" /> {lorasRoot || 'MODELS_PATH/loras'}
              </div>
              {filteredModels.length === 0 && (
                <div className="text-xs text-gray-500">No .safetensors files found under the loras folder.</div>
              )}
              <div className="space-y-0.5">
                {filteredModels.map(f => (
                  <button
                    type="button"
                    key={f.path}
                    onClick={() => pick(f)}
                    className="w-full flex items-center gap-2 px-2 py-1 rounded text-xs text-left hover:bg-blue-900/40"
                    title={f.path}
                  >
                    <span className="truncate flex-1 font-mono">{f.relpath || f.name}</span>
                    <span className="text-gray-500 shrink-0">{fmtSize(f.size)}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
}
