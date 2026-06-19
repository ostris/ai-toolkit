'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { Button } from '@headlessui/react';
import { Job } from '@prisma/client';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { CgSpinner } from 'react-icons/cg';
import { Download, Upload, GitCompare, Trash2, Pencil, Play } from 'lucide-react';
import classNames from 'classnames';

type FilterMode = 'all' | 'drafts' | 'past';

interface FlatRow {
  path: string;
  a: any;
  b: any;
  changed: boolean;
}

function flattenConfig(obj: any, prefix = ''): Record<string, any> {
  const out: Record<string, any> = {};
  if (obj === null || obj === undefined) {
    out[prefix || '(root)'] = obj;
    return out;
  }
  if (typeof obj !== 'object') {
    out[prefix || '(root)'] = obj;
    return out;
  }
  if (Array.isArray(obj)) {
    if (obj.length === 0) {
      out[prefix] = '[]';
      return out;
    }
    obj.forEach((v, i) => {
      Object.assign(out, flattenConfig(v, `${prefix}[${i}]`));
    });
    return out;
  }
  const keys = Object.keys(obj);
  if (keys.length === 0) {
    out[prefix] = '{}';
    return out;
  }
  keys.forEach(k => {
    const nextPrefix = prefix ? `${prefix}.${k}` : k;
    Object.assign(out, flattenConfig(obj[k], nextPrefix));
  });
  return out;
}

function buildCompareRows(a: any, b: any): FlatRow[] {
  const fa = flattenConfig(a);
  const fb = flattenConfig(b);
  const allKeys = Array.from(new Set([...Object.keys(fa), ...Object.keys(fb)])).sort();
  return allKeys.map(path => {
    const av = fa[path];
    const bv = fb[path];
    const changed = JSON.stringify(av) !== JSON.stringify(bv);
    return { path, a: av, b: bv, changed };
  });
}

function fmtVal(v: any): string {
  if (v === undefined) return '—';
  if (v === null) return 'null';
  if (typeof v === 'string') return v;
  return JSON.stringify(v);
}

function downloadJSON(filename: string, data: any) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function DraftJobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterMode>('drafts');
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<string[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const [onlyDiffs, setOnlyDiffs] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = () => {
    setLoading(true);
    apiClient
      .get('/api/jobs?status=draft,completed,stopped,failed')
      .then(res => {
        setJobs(res.data.jobs || []);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading jobs:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    refresh();
  }, []);

  const filtered = useMemo(() => {
    let rows = jobs;
    if (filter === 'drafts') rows = rows.filter(j => j.status === 'draft');
    else if (filter === 'past') rows = rows.filter(j => j.status !== 'draft');
    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter(j => j.name.toLowerCase().includes(q));
    }
    return rows;
  }, [jobs, filter, search]);

  const toggleSelect = (id: string) => {
    setSelected(prev => {
      if (prev.includes(id)) return prev.filter(x => x !== id);
      if (prev.length >= 2) return [prev[1], id];
      return [...prev, id];
    });
  };

  const selectedJobs = useMemo(
    () => selected.map(id => jobs.find(j => j.id === id)).filter(Boolean) as Job[],
    [selected, jobs],
  );

  const compareRows = useMemo(() => {
    if (selectedJobs.length !== 2) return [];
    try {
      const a = JSON.parse(selectedJobs[0].job_config);
      const b = JSON.parse(selectedJobs[1].job_config);
      const all = buildCompareRows(a, b);
      return onlyDiffs ? all.filter(r => r.changed) : all;
    } catch {
      return [];
    }
  }, [selectedJobs, onlyDiffs]);

  const handleExport = (job: Job) => {
    let config: any = {};
    try {
      config = JSON.parse(job.job_config);
    } catch {
      config = { raw: job.job_config };
    }
    const payload = {
      _aitk_draft_version: 1,
      name: job.name,
      gpu_ids: job.gpu_ids,
      exported_at: new Date().toISOString(),
      source_status: job.status,
      job_config: config,
    };
    const safeName = job.name.replace(/[^a-z0-9_-]+/gi, '_');
    downloadJSON(`${safeName}.draft.json`, payload);
  };

  const handleExportSelected = () => {
    if (selectedJobs.length === 0) return;
    if (selectedJobs.length === 1) {
      handleExport(selectedJobs[0]);
      return;
    }
    const payload = {
      _aitk_draft_version: 1,
      exported_at: new Date().toISOString(),
      drafts: selectedJobs.map(j => ({
        name: j.name,
        gpu_ids: j.gpu_ids,
        source_status: j.status,
        job_config: (() => {
          try {
            return JSON.parse(j.job_config);
          } catch {
            return { raw: j.job_config };
          }
        })(),
      })),
    };
    downloadJSON(`drafts_${Date.now()}.json`, payload);
  };

  const handleImportClick = () => fileInputRef.current?.click();

  const handleFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      const items: any[] = Array.isArray(parsed?.drafts)
        ? parsed.drafts
        : [parsed];

      for (const item of items) {
        const baseName = item.name || item?.job_config?.config?.name || `imported_${Date.now()}`;
        const job_config = item.job_config || item.config ? (item.job_config ?? item) : item;
        const gpu_ids = item.gpu_ids || '0';

        let candidateName = baseName;
        let attempt = 0;
        while (attempt < 50) {
          try {
            await apiClient.post('/api/jobs', {
              name: candidateName,
              gpu_ids,
              job_config,
              status: 'draft',
            });
            break;
          } catch (err: any) {
            if (err.response?.status === 409) {
              attempt += 1;
              candidateName = `${baseName}_imported_${attempt}`;
              continue;
            }
            throw err;
          }
        }
      }
      refresh();
    } catch (err) {
      console.error('Import failed:', err);
      alert('Failed to import file. Expected a JSON draft export.');
    }
  };

  const handleDelete = async (job: Job) => {
    if (!confirm(`Delete "${job.name}"? This cannot be undone.`)) return;
    try {
      await apiClient.get(`/api/jobs/${job.id}/delete`);
      setSelected(prev => prev.filter(id => id !== job.id));
      refresh();
    } catch (err) {
      console.error('Delete failed:', err);
      alert('Failed to delete job.');
    }
  };

  const handlePromote = async (job: Job) => {
    try {
      await apiClient.post('/api/jobs', {
        id: job.id,
        name: job.name,
        gpu_ids: job.gpu_ids,
        job_config: JSON.parse(job.job_config),
        status: 'stopped',
      });
      refresh();
      alert('Draft promoted. Find it in the Training Queue under Idle.');
    } catch (err) {
      console.error('Promote failed:', err);
      alert('Failed to promote draft.');
    }
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Draft Jobs</h1>
        </div>
        <div className="flex-1"></div>
        <div className="flex items-center gap-2 pr-2">
          <input
            type="text"
            placeholder="Search by name..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="bg-gray-800 text-gray-200 text-sm px-2 py-1 rounded-md border border-gray-700 focus:outline-none focus:border-gray-500"
          />
          <div className="flex bg-gray-800 rounded-md text-sm overflow-hidden border border-gray-700">
            {(['drafts', 'past', 'all'] as FilterMode[]).map(mode => (
              <button
                key={mode}
                onClick={() => setFilter(mode)}
                className={classNames('px-3 py-1 capitalize', {
                  'bg-slate-600 text-white': filter === mode,
                  'text-gray-300 hover:bg-gray-700': filter !== mode,
                })}
              >
                {mode}
              </button>
            ))}
          </div>
          <Button
            className="flex items-center gap-1 text-gray-200 bg-gray-800 hover:bg-gray-700 px-3 py-1 rounded-md text-sm"
            onClick={handleImportClick}
          >
            <Upload className="w-4 h-4" /> Import
          </Button>
          <Button
            className="flex items-center gap-1 text-gray-200 bg-gray-800 hover:bg-gray-700 px-3 py-1 rounded-md text-sm disabled:opacity-40"
            onClick={handleExportSelected}
            disabled={selectedJobs.length === 0}
          >
            <Download className="w-4 h-4" /> Export
            {selectedJobs.length > 0 ? ` (${selectedJobs.length})` : ''}
          </Button>
          <Button
            className="flex items-center gap-1 text-white bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded-md text-sm disabled:opacity-40"
            onClick={() => setCompareOpen(true)}
            disabled={selectedJobs.length !== 2}
          >
            <GitCompare className="w-4 h-4" /> Compare
          </Button>
          <Link
            href="/jobs/new"
            className="text-white bg-slate-600 hover:bg-slate-700 px-3 py-1 rounded-md text-sm"
          >
            New Draft
          </Link>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={handleFileSelected}
        />
      </TopBar>

      <MainContent>
        <div className="text-xs text-gray-400 mb-3">
          Select up to two jobs to compare their settings side-by-side. Drafts are saved
          configurations that aren't queued for training — promote a draft to push it into
          the Idle pool of the Training Queue.
        </div>

        {loading ? (
          <div className="flex items-center text-gray-400 py-8">
            <CgSpinner className="animate-spin mr-2" /> Loading...
          </div>
        ) : filtered.length === 0 ? (
          <div className="border border-dashed border-gray-700 rounded-lg p-8 text-center text-gray-400">
            No {filter === 'all' ? '' : filter} jobs found.{' '}
            <Link href="/jobs/new" className="text-blue-400 hover:underline">
              Create one
            </Link>{' '}
            or import a draft JSON file.
          </div>
        ) : (
          <div className="overflow-auto rounded-lg border border-gray-800">
            <table className="w-full text-sm">
              <thead className="bg-gray-800 text-gray-300">
                <tr>
                  <th className="px-3 py-2 text-left w-10"></th>
                  <th className="px-3 py-2 text-left">Name</th>
                  <th className="px-3 py-2 text-left">Status</th>
                  <th className="px-3 py-2 text-left">GPU</th>
                  <th className="px-3 py-2 text-left">Created</th>
                  <th className="px-3 py-2 text-left">Updated</th>
                  <th className="px-3 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(job => {
                  const isSel = selected.includes(job.id);
                  const isDraft = job.status === 'draft';
                  return (
                    <tr
                      key={job.id}
                      className={classNames(
                        'border-t border-gray-800 hover:bg-gray-900',
                        isSel && 'bg-blue-950/40',
                      )}
                    >
                      <td className="px-3 py-2">
                        <input
                          type="checkbox"
                          checked={isSel}
                          onChange={() => toggleSelect(job.id)}
                        />
                      </td>
                      <td className="px-3 py-2 font-medium">{job.name}</td>
                      <td className="px-3 py-2">
                        <span
                          className={classNames('text-xs px-2 py-0.5 rounded-full', {
                            'bg-amber-900/60 text-amber-300': isDraft,
                            'bg-green-900/60 text-green-300': job.status === 'completed',
                            'bg-red-900/60 text-red-300': job.status === 'failed',
                            'bg-gray-700 text-gray-300': !['draft', 'completed', 'failed'].includes(job.status),
                          })}
                        >
                          {job.status}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-gray-400">{job.gpu_ids}</td>
                      <td className="px-3 py-2 text-gray-400 whitespace-nowrap">
                        {new Date(job.created_at).toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-gray-400 whitespace-nowrap">
                        {new Date(job.updated_at).toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <div className="flex justify-end gap-1">
                          <Link
                            href={`/jobs/new?id=${job.id}`}
                            className="p-1 text-gray-300 hover:text-white hover:bg-gray-800 rounded"
                            title="Edit"
                          >
                            <Pencil className="w-4 h-4" />
                          </Link>
                          <button
                            onClick={() => handleExport(job)}
                            className="p-1 text-gray-300 hover:text-white hover:bg-gray-800 rounded"
                            title="Export this job"
                          >
                            <Download className="w-4 h-4" />
                          </button>
                          {isDraft && (
                            <button
                              onClick={() => handlePromote(job)}
                              className="p-1 text-green-400 hover:text-green-300 hover:bg-gray-800 rounded"
                              title="Promote to queue"
                            >
                              <Play className="w-4 h-4" />
                            </button>
                          )}
                          <button
                            onClick={() => handleDelete(job)}
                            className="p-1 text-red-400 hover:text-red-300 hover:bg-gray-800 rounded"
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {compareOpen && selectedJobs.length === 2 && (
          <div
            className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4"
            onClick={() => setCompareOpen(false)}
          >
            <div
              className="bg-gray-900 border border-gray-700 rounded-lg w-[95vw] max-w-6xl max-h-[90vh] flex flex-col"
              onClick={e => e.stopPropagation()}
            >
              <div className="flex items-center px-4 py-3 border-b border-gray-800">
                <div className="text-lg flex-1">Compare Settings</div>
                <label className="text-xs text-gray-300 mr-4 flex items-center gap-1">
                  <input
                    type="checkbox"
                    checked={onlyDiffs}
                    onChange={e => setOnlyDiffs(e.target.checked)}
                  />
                  Only show differences
                </label>
                <button
                  className="text-gray-300 hover:text-white"
                  onClick={() => setCompareOpen(false)}
                >
                  ✕
                </button>
              </div>
              <div className="overflow-auto flex-1">
                <table className="w-full text-xs font-mono">
                  <thead className="bg-gray-800 text-gray-300 sticky top-0">
                    <tr>
                      <th className="px-3 py-2 text-left">Setting</th>
                      <th className="px-3 py-2 text-left">{selectedJobs[0].name}</th>
                      <th className="px-3 py-2 text-left">{selectedJobs[1].name}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compareRows.length === 0 ? (
                      <tr>
                        <td colSpan={3} className="text-center text-gray-400 py-8">
                          {onlyDiffs ? 'No differences found.' : 'No fields.'}
                        </td>
                      </tr>
                    ) : (
                      compareRows.map(row => (
                        <tr
                          key={row.path}
                          className={classNames('border-t border-gray-800', {
                            'bg-yellow-900/20': row.changed,
                          })}
                        >
                          <td className="px-3 py-1 text-gray-300 align-top">{row.path}</td>
                          <td
                            className={classNames('px-3 py-1 align-top whitespace-pre-wrap break-all', {
                              'text-red-300': row.changed,
                              'text-gray-400': !row.changed,
                            })}
                          >
                            {fmtVal(row.a)}
                          </td>
                          <td
                            className={classNames('px-3 py-1 align-top whitespace-pre-wrap break-all', {
                              'text-green-300': row.changed,
                              'text-gray-400': !row.changed,
                            })}
                          >
                            {fmtVal(row.b)}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </MainContent>
    </>
  );
}
