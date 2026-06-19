'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { Plus, Trash2, Pencil, Save, X, Download, Upload, Search, Copy } from 'lucide-react';
import { modelArchs } from '@/app/jobs/new/options';
import { CONFIG_PRESETS } from '@/app/jobs/new/configPresets';
import classNames from 'classnames';

interface UserPreset {
  id: string;
  name: string;
  description: string;
  modelArchs: string[];
  overrides: Record<string, any>;
  approxVramGB?: number;
  tier?: string;
  created_at: string;
  updated_at: string;
  source?: 'user' | 'builtin';
}

function flattenBuiltins(): UserPreset[] {
  const out: UserPreset[] = [];
  for (const [archName, list] of Object.entries(CONFIG_PRESETS)) {
    for (const p of list) {
      out.push({
        id: `builtin:${archName}:${p.id}`,
        name: p.label,
        description: p.description,
        modelArchs: [archName],
        overrides: p.overrides as Record<string, any>,
        approxVramGB: p.approxVramGB,
        tier: p.tier,
        created_at: '',
        updated_at: '',
        source: 'builtin',
      });
    }
  }
  return out;
}

function uid() {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

function normalize(raw: any): UserPreset {
  return {
    id: raw.id || uid(),
    name: raw.name || 'Untitled preset',
    description: raw.description || '',
    modelArchs: Array.isArray(raw.modelArchs) ? raw.modelArchs.filter((x: any) => typeof x === 'string') : [],
    overrides: raw.overrides && typeof raw.overrides === 'object' ? raw.overrides : {},
    approxVramGB: typeof raw.approxVramGB === 'number' ? raw.approxVramGB : undefined,
    created_at: raw.created_at || new Date().toISOString(),
    updated_at: raw.updated_at || new Date().toISOString(),
  };
}

const overridesPlaceholder = `{
  "config.process[0].train.batch_size": 1,
  "config.process[0].train.gradient_accumulation": 4,
  "config.process[0].datasets[0].resolution": [512, 768]
}`;

export default function PresetsPage() {
  const [presets, setPresets] = useState<UserPreset[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState('');
  const [archFilter, setArchFilter] = useState<string>('all');
  const [showBuiltins, setShowBuiltins] = useState(true);
  const builtins = useMemo(() => flattenBuiltins(), []);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState<UserPreset | null>(null);
  const [overridesText, setOverridesText] = useState('');
  const [overridesError, setOverridesError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const skipNextSave = useRef(true);

  useEffect(() => {
    apiClient
      .get('/api/presets')
      .then(res => {
        setPresets((res.data?.presets || []).map(normalize));
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (loading) return;
    if (skipNextSave.current) {
      skipNextSave.current = false;
      return;
    }
    setSaving(true);
    const t = setTimeout(() => {
      apiClient
        .post('/api/presets', { presets })
        .catch(err => console.error('Failed to save presets:', err))
        .finally(() => setSaving(false));
    }, 500);
    return () => clearTimeout(t);
  }, [presets, loading]);

  const archOptions = useMemo(
    () => modelArchs.map(a => ({ value: a.name, label: a.label })),
    [],
  );

  const filter = (rows: UserPreset[]) => {
    let out = rows;
    if (archFilter !== 'all') out = out.filter(p => p.modelArchs.includes(archFilter));
    if (search.trim()) {
      const q = search.toLowerCase();
      out = out.filter(
        p =>
          p.name.toLowerCase().includes(q) ||
          p.description.toLowerCase().includes(q) ||
          p.modelArchs.some(a => a.toLowerCase().includes(q)),
      );
    }
    return out;
  };

  const filteredUser = useMemo(() => filter(presets), [presets, archFilter, search]);
  const filteredBuiltins = useMemo(
    () => (showBuiltins ? filter(builtins) : []),
    [builtins, archFilter, search, showBuiltins],
  );

  const startEdit = (p: UserPreset) => {
    setEditingId(p.id);
    setDraft({ ...p, modelArchs: [...p.modelArchs], overrides: { ...p.overrides } });
    setOverridesText(JSON.stringify(p.overrides, null, 2));
    setOverridesError(null);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setDraft(null);
    setOverridesError(null);
  };

  const saveEdit = () => {
    if (!draft) return;
    let parsedOverrides: Record<string, any>;
    try {
      parsedOverrides = JSON.parse(overridesText || '{}');
      if (typeof parsedOverrides !== 'object' || Array.isArray(parsedOverrides) || parsedOverrides === null) {
        throw new Error('Overrides must be a JSON object.');
      }
    } catch (err: any) {
      setOverridesError(err.message || 'Invalid JSON');
      return;
    }
    const updated: UserPreset = {
      ...draft,
      overrides: parsedOverrides,
      updated_at: new Date().toISOString(),
    };
    setPresets(prev => (prev.some(p => p.id === draft.id) ? prev.map(p => (p.id === draft.id ? updated : p)) : [updated, ...prev]));
    cancelEdit();
  };

  const addPreset = () => {
    const now = new Date().toISOString();
    const p: UserPreset = {
      id: uid(),
      name: 'New preset',
      description: '',
      modelArchs: archFilter !== 'all' ? [archFilter] : [],
      overrides: {},
      created_at: now,
      updated_at: now,
    };
    setPresets(prev => [p, ...prev]);
    startEdit(p);
  };

  const duplicatePreset = (p: UserPreset) => {
    const now = new Date().toISOString();
    const isBuiltin = p.source === 'builtin';
    const copy: UserPreset = {
      ...p,
      id: uid(),
      name: isBuiltin ? `${p.name} (custom)` : `${p.name} (copy)`,
      modelArchs: [...p.modelArchs],
      overrides: { ...p.overrides },
      source: 'user',
      created_at: now,
      updated_at: now,
    };
    setPresets(prev => [copy, ...prev]);
    if (isBuiltin) startEdit(copy);
  };

  const deletePreset = (id: string) => {
    if (!confirm('Delete this preset?')) return;
    setPresets(prev => prev.filter(p => p.id !== id));
    if (editingId === id) cancelEdit();
  };

  const toggleArch = (archName: string) => {
    if (!draft) return;
    setDraft({
      ...draft,
      modelArchs: draft.modelArchs.includes(archName)
        ? draft.modelArchs.filter(a => a !== archName)
        : [...draft.modelArchs, archName],
    });
  };

  const handleExport = () => {
    const blob = new Blob([JSON.stringify({ presets }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `preset_configurations_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleImportClick = () => fileInputRef.current?.click();

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const items: UserPreset[] = (parsed.presets || (Array.isArray(parsed) ? parsed : [parsed])).map(normalize);
      items.forEach(item => (item.id = uid()));
      setPresets(prev => [...items, ...prev]);
    } catch (err) {
      console.error('Import failed:', err);
      alert('Failed to import — expected a JSON file with a "presets" array.');
    }
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Preset Configurations</h1>
        </div>
        <div className="flex-1"></div>
        <div className="flex items-center gap-2 pr-2">
          <span className="text-xs text-gray-400">
            {saving ? 'Saving...' : `${presets.length} presets`}
          </span>
          <Button
            className="flex items-center gap-1 text-gray-200 bg-gray-800 hover:bg-gray-700 px-3 py-1 rounded-md text-sm"
            onClick={handleImportClick}
          >
            <Upload className="w-4 h-4" /> Import
          </Button>
          <Button
            className="flex items-center gap-1 text-gray-200 bg-gray-800 hover:bg-gray-700 px-3 py-1 rounded-md text-sm"
            onClick={handleExport}
          >
            <Download className="w-4 h-4" /> Export
          </Button>
          <Button
            className="flex items-center gap-1 text-white bg-green-600 hover:bg-green-700 px-3 py-1 rounded-md text-sm"
            onClick={addPreset}
          >
            <Plus className="w-4 h-4" /> New Preset
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={handleFile}
          />
        </div>
      </TopBar>

      <MainContent>
        <div className="flex items-center gap-2 mb-3">
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 absolute left-2 top-2 text-gray-500" />
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search presets..."
              className="w-full bg-gray-900 border border-gray-700 rounded pl-8 pr-3 py-1.5 text-sm text-gray-200"
            />
          </div>
          <select
            value={archFilter}
            onChange={e => setArchFilter(e.target.value)}
            className="bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200"
          >
            <option value="all">All models</option>
            {archOptions.map(o => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1 text-xs text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={showBuiltins}
              onChange={e => setShowBuiltins(e.target.checked)}
            />
            Show built-in presets ({builtins.length})
          </label>
        </div>

        <div className="text-xs text-gray-400 mb-3">
          User-defined presets show up alongside the built-in suggestions on the New Job page for
          every model architecture you assign here. Overrides use the same dotted-path format as a
          JobConfig — e.g. <code className="bg-gray-800 px-1 rounded">config.process[0].train.batch_size</code>.
        </div>

        {loading ? (
          <div className="text-gray-400">Loading...</div>
        ) : filteredUser.length === 0 && filteredBuiltins.length === 0 ? (
          <div className="border border-dashed border-gray-700 rounded-lg p-8 text-center text-gray-400">
            No presets match the current filter.
          </div>
        ) : (
          <div className="space-y-3 pb-12">
            {filteredUser.length > 0 && (
              <div className="text-xs uppercase tracking-wide text-gray-500 pb-1">
                Your presets ({filteredUser.length})
              </div>
            )}
            {filteredUser.length === 0 && showBuiltins && (
              <div className="text-xs uppercase tracking-wide text-gray-500 pb-1">
                Your presets (none yet — customize a built-in or create one)
              </div>
            )}
            {[...filteredUser, ...(filteredBuiltins.length > 0 ? [{ id: '__divider__' } as any] : []), ...filteredBuiltins].map((p: any) => {
              if (p.id === '__divider__') {
                return (
                  <div
                    key="__divider__"
                    className="text-xs uppercase tracking-wide text-gray-500 pt-4 pb-1 border-t border-gray-800 mt-4"
                  >
                    Built-in presets ({filteredBuiltins.length}) — read-only; click Customize to make an editable copy
                  </div>
                );
              }
              const isBuiltin = p.source === 'builtin';
              const isEditing = editingId === p.id;
              if (isEditing && draft) {
                return (
                  <div key={p.id} className="border border-blue-700 bg-gray-900 rounded-lg p-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                      <div>
                        <div className="text-xs text-gray-400 mb-1">Name</div>
                        <input
                          type="text"
                          value={draft.name}
                          onChange={e => setDraft({ ...draft, name: e.target.value })}
                          className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100"
                        />
                      </div>
                      <div>
                        <div className="text-xs text-gray-400 mb-1">Approx VRAM (GB, optional)</div>
                        <input
                          type="text"
                          inputMode="numeric"
                          value={draft.approxVramGB ?? ''}
                          onChange={e => {
                            const v = e.target.value.replace(/[^0-9.]/g, '');
                            setDraft({ ...draft, approxVramGB: v === '' ? undefined : Number(v) });
                          }}
                          className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100"
                        />
                      </div>
                    </div>
                    <div className="mb-3">
                      <div className="text-xs text-gray-400 mb-1">Description</div>
                      <input
                        type="text"
                        value={draft.description}
                        onChange={e => setDraft({ ...draft, description: e.target.value })}
                        placeholder="One-line description shown on the card"
                        className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100"
                      />
                    </div>
                    <div className="mb-3">
                      <div className="text-xs text-gray-400 mb-1">Apply to models ({draft.modelArchs.length})</div>
                      <div className="border border-gray-700 rounded p-2 max-h-48 overflow-auto bg-gray-950">
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-1">
                          {archOptions.map(a => (
                            <label
                              key={a.value}
                              className="flex items-center gap-1 text-xs text-gray-300 hover:bg-gray-800 px-1 py-0.5 rounded cursor-pointer"
                            >
                              <input
                                type="checkbox"
                                checked={draft.modelArchs.includes(a.value)}
                                onChange={() => toggleArch(a.value)}
                              />
                              {a.label}
                            </label>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="mb-3">
                      <div className="text-xs text-gray-400 mb-1">
                        Overrides (JSON: dotted path → value)
                      </div>
                      <textarea
                        value={overridesText}
                        onChange={e => {
                          setOverridesText(e.target.value);
                          setOverridesError(null);
                        }}
                        rows={10}
                        placeholder={overridesPlaceholder}
                        className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 font-mono"
                      />
                      {overridesError && (
                        <div className="text-xs text-red-400 mt-1">{overridesError}</div>
                      )}
                    </div>
                    <div className="flex justify-end gap-2">
                      <button
                        onClick={cancelEdit}
                        className="px-3 py-1 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={saveEdit}
                        className="px-3 py-1 text-sm text-white bg-green-600 hover:bg-green-700 rounded flex items-center gap-1"
                      >
                        <Save className="w-3 h-3" /> Save
                      </button>
                    </div>
                  </div>
                );
              }
              const overrideCount = Object.keys(p.overrides).length;
              return (
                <div
                  key={p.id}
                  className={classNames(
                    'group border rounded-lg p-4',
                    isBuiltin
                      ? 'border-cyan-900/60 bg-cyan-950/10'
                      : 'border-gray-800 hover:border-gray-700 bg-gray-900/50',
                  )}
                >
                  <div className="flex items-start gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <div className="text-sm font-medium text-gray-100">{p.name}</div>
                        {isBuiltin && (
                          <span className="text-[10px] uppercase tracking-wide bg-cyan-700 text-white px-1.5 py-0.5 rounded">
                            Built-in
                          </span>
                        )}
                        {p.tier && isBuiltin && (
                          <span className="text-[10px] uppercase tracking-wide bg-gray-700 text-gray-300 px-1.5 py-0.5 rounded">
                            {p.tier}
                          </span>
                        )}
                        {p.approxVramGB !== undefined && p.approxVramGB > 0 && (
                          <span className="text-[11px] bg-cyan-900/40 text-cyan-300 px-1.5 py-0.5 rounded">
                            ~{p.approxVramGB} GB
                          </span>
                        )}
                      </div>
                      {p.description && (
                        <div className="text-xs text-gray-400 mb-2">{p.description}</div>
                      )}
                      <div className="flex flex-wrap gap-1 mb-2">
                        {p.modelArchs.length === 0 ? (
                          <span className="text-[11px] text-amber-400 italic">
                            No models assigned — won't appear on any job.
                          </span>
                        ) : (
                          p.modelArchs.map(arch => {
                            const meta = archOptions.find(o => o.value === arch);
                            return (
                              <span
                                key={arch}
                                className="text-[11px] bg-gray-800 text-gray-300 px-2 py-0.5 rounded-full"
                              >
                                {meta?.label || arch}
                              </span>
                            );
                          })
                        )}
                      </div>
                      <div className="text-[11px] text-gray-500">
                        {overrideCount} override{overrideCount === 1 ? '' : 's'}
                      </div>
                    </div>
                    {isBuiltin ? (
                      <div className="flex gap-1">
                        <button
                          onClick={() => duplicatePreset(p)}
                          className="flex items-center gap-1 px-2 py-1 text-xs bg-cyan-800 hover:bg-cyan-700 text-white rounded"
                          title="Make an editable copy in your library"
                        >
                          <Copy className="w-3 h-3" /> Customize
                        </button>
                      </div>
                    ) : (
                      <div className="flex gap-1 opacity-0 group-hover:opacity-100">
                        <button
                          onClick={() => duplicatePreset(p)}
                          className="p-1 text-gray-300 hover:text-white hover:bg-gray-800 rounded"
                          title="Duplicate"
                        >
                          <Copy className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => startEdit(p)}
                          className="p-1 text-gray-300 hover:text-white hover:bg-gray-800 rounded"
                          title="Edit"
                        >
                          <Pencil className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => deletePreset(p.id)}
                          className="p-1 text-red-400 hover:text-red-300 hover:bg-gray-800 rounded"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </MainContent>
    </>
  );
}
