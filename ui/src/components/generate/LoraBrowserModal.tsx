'use client';

import { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Cloud, FolderOpen, Loader2, Search } from 'lucide-react';
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

type Tab = 'cloud' | 'jobs' | 'models';

const fmtSize = (n: number) => (n > 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${(n / 1e6).toFixed(0)} MB`);
const fmtDate = (ms: number) => new Date(ms).toLocaleString();

export default function LoraBrowserModal({ isOpen, onClose, onPick, cloudLoras }: Props) {
  const hasCloud = !!cloudLoras?.length;
  const [tab, setTab] = useState<Tab>(hasCloud ? 'cloud' : 'jobs');
  const [loading, setLoading] = useState(false);
  const [jobs, setJobs] = useState<JobEntry[]>([]);
  const [models, setModels] = useState<LoraFile[]>([]);
  const [lorasRoot, setLorasRoot] = useState('');
  const [filter, setFilter] = useState('');
  const [open, setOpen] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!isOpen) return;
    setTab(hasCloud ? 'cloud' : 'jobs');
    setLoading(true);
    apiClient
      .get('/api/loras')
      .then(r => {
        setJobs(r.data.jobs || []);
        setModels(r.data.models || []);
        setLorasRoot(r.data.lorasRoot || '');
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [isOpen]);

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
    `px-3 py-1.5 text-sm border-b-2 ${tab === t ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-400 hover:text-gray-200'}`;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Add LoRA" size="lg">
      <div className="text-gray-200">
        <div className="flex items-center gap-2 border-b border-gray-800 mb-3">
          {hasCloud && (
            <button type="button" className={tabClass('cloud')} onClick={() => setTab('cloud')}>
              For this model
            </button>
          )}
          <button type="button" className={tabClass('jobs')} onClick={() => setTab('jobs')}>
            Training jobs
          </button>
          <button type="button" className={tabClass('models')} onClick={() => setTab('models')}>
            Models folder
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
