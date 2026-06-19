'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { Plus, Trash2, Pencil, Save, X, Tag, Folder, Download, Upload, Search, Info, Sparkles } from 'lucide-react';
import GeneratePromptsModal, { GeneratedPromptDraft } from './GeneratePromptsModal';
import classNames from 'classnames';

interface PromptItem {
  id: string;
  text: string;
  title?: string;
  tags: string[];
  setIds: string[];
  notes?: string;
  width?: number;
  height?: number;
  seed?: number;
  network_multiplier?: number;
  created_at: string;
  updated_at: string;
}

const SEED_HELP =
  'Random seed used to initialize the noise for this prompt. The same seed + prompt + model produces the same sample every time, so reusing a seed lets you compare changes side-by-side and reproduce a result you liked. Leave blank to inherit the job-level seed (or "walk seed", which steps the seed by +1 for each prompt).';

const LORA_HELP =
  'How strongly the trained LoRA influences this sample (also called network multiplier or LoRA strength). 1.0 is the trained strength; 0 disables the LoRA entirely (base model only); >1 over-applies it and often produces artifacts. Use lower values (0.5–0.8) to test how much of the effect comes from training vs. the base model. Leave blank for the default of 1.0.';

interface PromptSet {
  id: string;
  name: string;
  description?: string;
  color?: string;
}

interface Library {
  prompts: PromptItem[];
  sets: PromptSet[];
}

const SET_COLORS = ['#60a5fa', '#34d399', '#fbbf24', '#f472b6', '#a78bfa', '#fb7185', '#22d3ee', '#facc15'];

function uid() {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

function normalizePrompt(raw: any): PromptItem {
  const numOrUndef = (v: any) => {
    if (v === '' || v === null || v === undefined) return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  };
  return {
    id: raw.id || uid(),
    text: typeof raw.text === 'string' ? raw.text : '',
    title: raw.title || '',
    tags: Array.isArray(raw.tags) ? raw.tags.filter((t: any) => typeof t === 'string') : [],
    setIds: Array.isArray(raw.setIds) ? raw.setIds.filter((s: any) => typeof s === 'string') : [],
    notes: raw.notes || '',
    width: numOrUndef(raw.width),
    height: numOrUndef(raw.height),
    seed: numOrUndef(raw.seed),
    network_multiplier: numOrUndef(raw.network_multiplier),
    created_at: raw.created_at || new Date().toISOString(),
    updated_at: raw.updated_at || new Date().toISOString(),
  };
}

function normalizeSet(raw: any): PromptSet {
  return {
    id: raw.id || uid(),
    name: raw.name || 'Untitled set',
    description: raw.description || '',
    color: raw.color || SET_COLORS[0],
  };
}

interface NumFieldProps {
  label: string;
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  placeholder?: string;
  help?: string;
  step?: string;
  allowDecimal?: boolean;
}

function NumField({ label, value, onChange, placeholder, help, step, allowDecimal }: NumFieldProps) {
  return (
    <div>
      <div className="text-xs text-gray-400 mb-1 flex items-center gap-1">
        {label}
        {help && (
          <span className="relative group">
            <Info className="w-3 h-3 text-gray-500 hover:text-gray-300 cursor-help" />
            <span
              className="pointer-events-none absolute left-0 top-5 z-50 hidden group-hover:block
                         w-72 bg-gray-950 border border-gray-700 text-gray-200 text-xs
                         rounded-md p-2 shadow-lg whitespace-normal leading-snug"
            >
              {help}
            </span>
          </span>
        )}
      </div>
      <input
        type="text"
        inputMode={allowDecimal ? 'decimal' : 'numeric'}
        step={step}
        value={value === undefined || value === null ? '' : `${value}`}
        placeholder={placeholder}
        onChange={e => {
          const raw = e.target.value;
          if (raw === '') {
            onChange(undefined);
            return;
          }
          const cleaned = allowDecimal ? raw.replace(/[^0-9.\-]/g, '') : raw.replace(/[^0-9]/g, '');
          if (cleaned === '' || cleaned === '-' || cleaned === '.' || cleaned === '-.') {
            onChange(undefined);
            return;
          }
          const n = Number(cleaned);
          if (Number.isFinite(n)) onChange(n);
        }}
        className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100"
      />
    </div>
  );
}

export default function PromptBuilderPage() {
  const [library, setLibrary] = useState<Library>({ prompts: [], sets: [] });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState('');
  const [activeSetId, setActiveSetId] = useState<string | 'all' | 'untagged'>('all');
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState<PromptItem | null>(null);
  const [editingSetId, setEditingSetId] = useState<string | null>(null);
  const [newSetName, setNewSetName] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const skipNextSave = useRef(true);
  const [generateOpen, setGenerateOpen] = useState(false);

  const handleGenerated = (drafts: GeneratedPromptDraft[]) => {
    const now = new Date().toISOString();
    const newPrompts: PromptItem[] = drafts.map(d => ({
      id: uid(),
      text: d.text,
      title: d.title || '',
      tags: d.tags || [],
      setIds: d.setIds || [],
      notes: '',
      width: d.width,
      height: d.height,
      seed: d.seed,
      network_multiplier: d.network_multiplier,
      created_at: now,
      updated_at: now,
    }));
    setLibrary(lib => ({ ...lib, prompts: [...newPrompts, ...lib.prompts] }));
  };

  useEffect(() => {
    apiClient
      .get('/api/prompts')
      .then(res => {
        const data = res.data || {};
        setLibrary({
          prompts: (data.prompts || []).map(normalizePrompt),
          sets: (data.sets || []).map(normalizeSet),
        });
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load prompts:', err);
        setLoading(false);
      });
  }, []);

  // Debounced autosave
  useEffect(() => {
    if (loading) return;
    if (skipNextSave.current) {
      skipNextSave.current = false;
      return;
    }
    setSaving(true);
    const t = setTimeout(() => {
      apiClient
        .post('/api/prompts', library)
        .catch(err => console.error('Failed to save library:', err))
        .finally(() => setSaving(false));
    }, 500);
    return () => clearTimeout(t);
  }, [library, loading]);

  const allTags = useMemo(() => {
    const tags = new Set<string>();
    library.prompts.forEach(p => p.tags.forEach(t => tags.add(t)));
    return Array.from(tags).sort();
  }, [library.prompts]);

  const filtered = useMemo(() => {
    let rows = library.prompts;
    if (activeSetId === 'untagged') {
      rows = rows.filter(p => p.setIds.length === 0);
    } else if (activeSetId !== 'all') {
      rows = rows.filter(p => p.setIds.includes(activeSetId));
    }
    if (activeTag) {
      rows = rows.filter(p => p.tags.includes(activeTag));
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter(
        p =>
          p.text.toLowerCase().includes(q) ||
          (p.title || '').toLowerCase().includes(q) ||
          p.tags.some(t => t.toLowerCase().includes(q)),
      );
    }
    return rows;
  }, [library.prompts, activeSetId, activeTag, search]);

  const startEdit = (prompt: PromptItem) => {
    setEditingId(prompt.id);
    setDraft({ ...prompt, tags: [...prompt.tags], setIds: [...prompt.setIds] });
  };

  const cancelEdit = () => {
    setEditingId(null);
    setDraft(null);
  };

  const saveEdit = () => {
    if (!draft) return;
    const updated = { ...draft, updated_at: new Date().toISOString() };
    setLibrary(lib => ({
      ...lib,
      prompts: lib.prompts.map(p => (p.id === draft.id ? updated : p)),
    }));
    cancelEdit();
  };

  const addPrompt = () => {
    const now = new Date().toISOString();
    const newPrompt: PromptItem = {
      id: uid(),
      text: '',
      title: '',
      tags: [],
      setIds: activeSetId !== 'all' && activeSetId !== 'untagged' ? [activeSetId] : [],
      notes: '',
      created_at: now,
      updated_at: now,
    };
    setLibrary(lib => ({ ...lib, prompts: [newPrompt, ...lib.prompts] }));
    startEdit(newPrompt);
  };

  const deletePrompt = (id: string) => {
    if (!confirm('Delete this prompt?')) return;
    setLibrary(lib => ({ ...lib, prompts: lib.prompts.filter(p => p.id !== id) }));
    if (editingId === id) cancelEdit();
  };

  const addSet = () => {
    const name = newSetName.trim();
    if (!name) return;
    const newSet: PromptSet = {
      id: uid(),
      name,
      description: '',
      color: SET_COLORS[library.sets.length % SET_COLORS.length],
    };
    setLibrary(lib => ({ ...lib, sets: [...lib.sets, newSet] }));
    setNewSetName('');
  };

  const renameSet = (id: string, name: string) => {
    setLibrary(lib => ({
      ...lib,
      sets: lib.sets.map(s => (s.id === id ? { ...s, name } : s)),
    }));
  };

  const deleteSet = (id: string) => {
    const set = library.sets.find(s => s.id === id);
    if (!set) return;
    if (!confirm(`Delete set "${set.name}"? Prompts in it will not be deleted.`)) return;
    setLibrary(lib => ({
      ...lib,
      sets: lib.sets.filter(s => s.id !== id),
      prompts: lib.prompts.map(p => ({ ...p, setIds: p.setIds.filter(sid => sid !== id) })),
    }));
    if (activeSetId === id) setActiveSetId('all');
  };

  const toggleDraftTag = (tag: string) => {
    if (!draft) return;
    setDraft({
      ...draft,
      tags: draft.tags.includes(tag) ? draft.tags.filter(t => t !== tag) : [...draft.tags, tag],
    });
  };

  const toggleDraftSet = (id: string) => {
    if (!draft) return;
    setDraft({
      ...draft,
      setIds: draft.setIds.includes(id) ? draft.setIds.filter(s => s !== id) : [...draft.setIds, id],
    });
  };

  const addDraftTagFromInput = (val: string) => {
    if (!draft) return;
    const t = val.trim();
    if (!t || draft.tags.includes(t)) return;
    setDraft({ ...draft, tags: [...draft.tags, t] });
  };

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(library, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prompt_library_${Date.now()}.json`;
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
      const text = await file.text();
      const parsed = JSON.parse(text);
      const imported: Library = {
        prompts: (parsed.prompts || []).map(normalizePrompt),
        sets: (parsed.sets || []).map(normalizeSet),
      };
      if (
        library.prompts.length === 0 ||
        confirm(`Merge ${imported.prompts.length} prompts and ${imported.sets.length} sets into your library?`)
      ) {
        setLibrary(lib => ({
          prompts: [...lib.prompts, ...imported.prompts],
          sets: [...lib.sets, ...imported.sets],
        }));
      }
    } catch (err) {
      console.error('Import failed:', err);
      alert('Failed to import — expected a JSON file exported from the prompt library.');
    }
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Prompt Builder</h1>
        </div>
        <div className="flex-1"></div>
        <div className="flex items-center gap-2 pr-2">
          <span className="text-xs text-gray-400">
            {saving ? 'Saving...' : `${library.prompts.length} prompts, ${library.sets.length} sets`}
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
            className="flex items-center gap-1 text-white bg-amber-600 hover:bg-amber-500 px-3 py-1 rounded-md text-sm"
            onClick={() => setGenerateOpen(true)}
          >
            <Sparkles className="w-4 h-4" /> Generate Prompts
          </Button>
          <Button
            className="flex items-center gap-1 text-white bg-green-600 hover:bg-green-700 px-3 py-1 rounded-md text-sm"
            onClick={addPrompt}
          >
            <Plus className="w-4 h-4" /> New Prompt
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
        <div className="flex gap-4 h-[calc(100vh-80px)]">
          {/* Sets sidebar */}
          <div className="w-64 flex-shrink-0 border border-gray-800 rounded-lg p-3 overflow-auto">
            <div className="text-xs uppercase text-gray-400 mb-2 flex items-center gap-1">
              <Folder className="w-3 h-3" /> Sets
            </div>
            <button
              onClick={() => setActiveSetId('all')}
              className={classNames('w-full text-left px-2 py-1 rounded text-sm mb-1', {
                'bg-blue-900/40 text-blue-200': activeSetId === 'all',
                'text-gray-300 hover:bg-gray-800': activeSetId !== 'all',
              })}
            >
              All Prompts <span className="text-gray-500">({library.prompts.length})</span>
            </button>
            <button
              onClick={() => setActiveSetId('untagged')}
              className={classNames('w-full text-left px-2 py-1 rounded text-sm mb-2', {
                'bg-blue-900/40 text-blue-200': activeSetId === 'untagged',
                'text-gray-300 hover:bg-gray-800': activeSetId !== 'untagged',
              })}
            >
              Uncategorized{' '}
              <span className="text-gray-500">
                ({library.prompts.filter(p => p.setIds.length === 0).length})
              </span>
            </button>
            <div className="border-t border-gray-800 my-2"></div>
            {library.sets.map(set => {
              const count = library.prompts.filter(p => p.setIds.includes(set.id)).length;
              const isEditing = editingSetId === set.id;
              return (
                <div
                  key={set.id}
                  className={classNames(
                    'group flex items-center gap-1 px-2 py-1 rounded text-sm mb-1',
                    activeSetId === set.id ? 'bg-blue-900/40' : 'hover:bg-gray-800',
                  )}
                >
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: set.color }}
                  />
                  {isEditing ? (
                    <input
                      autoFocus
                      defaultValue={set.name}
                      onBlur={e => {
                        renameSet(set.id, e.target.value || set.name);
                        setEditingSetId(null);
                      }}
                      onKeyDown={e => {
                        if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                        if (e.key === 'Escape') setEditingSetId(null);
                      }}
                      className="flex-1 bg-gray-900 text-gray-200 px-1 rounded text-sm border border-gray-600"
                    />
                  ) : (
                    <button
                      onClick={() => setActiveSetId(set.id)}
                      className="flex-1 text-left text-gray-300 truncate"
                    >
                      {set.name} <span className="text-gray-500">({count})</span>
                    </button>
                  )}
                  <button
                    onClick={() => setEditingSetId(set.id)}
                    className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-white"
                    title="Rename"
                  >
                    <Pencil className="w-3 h-3" />
                  </button>
                  <button
                    onClick={() => deleteSet(set.id)}
                    className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300"
                    title="Delete set"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              );
            })}
            <div className="mt-3 flex gap-1">
              <input
                type="text"
                value={newSetName}
                onChange={e => setNewSetName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && addSet()}
                placeholder="New set name"
                className="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200"
              />
              <button
                onClick={addSet}
                className="px-2 bg-gray-700 hover:bg-gray-600 rounded text-gray-200"
                title="Add set"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>

            {allTags.length > 0 && (
              <>
                <div className="text-xs uppercase text-gray-400 mt-4 mb-2 flex items-center gap-1">
                  <Tag className="w-3 h-3" /> Tags
                </div>
                <div className="flex flex-wrap gap-1">
                  {allTags.map(tag => (
                    <button
                      key={tag}
                      onClick={() => setActiveTag(activeTag === tag ? null : tag)}
                      className={classNames('text-xs px-2 py-0.5 rounded-full', {
                        'bg-blue-600 text-white': activeTag === tag,
                        'bg-gray-800 text-gray-300 hover:bg-gray-700': activeTag !== tag,
                      })}
                    >
                      {tag}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Prompts list */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="flex items-center gap-2 mb-3">
              <div className="relative flex-1">
                <Search className="w-4 h-4 absolute left-2 top-2 text-gray-500" />
                <input
                  type="text"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Search prompts, tags, or titles..."
                  className="w-full bg-gray-900 border border-gray-700 rounded pl-8 pr-3 py-1.5 text-sm text-gray-200"
                />
              </div>
              {activeTag && (
                <button
                  onClick={() => setActiveTag(null)}
                  className="text-xs bg-blue-900/40 text-blue-200 px-2 py-1 rounded flex items-center gap-1"
                >
                  Tag: {activeTag} <X className="w-3 h-3" />
                </button>
              )}
            </div>

            <div className="flex-1 overflow-auto pr-1 space-y-2">
              {loading ? (
                <div className="text-gray-400">Loading...</div>
              ) : filtered.length === 0 ? (
                <div className="border border-dashed border-gray-700 rounded-lg p-8 text-center text-gray-400">
                  No prompts here yet. Click <span className="text-green-400">New Prompt</span> to add one.
                </div>
              ) : (
                filtered.map(p => {
                  const isEditing = editingId === p.id;
                  if (isEditing && draft) {
                    return (
                      <div key={p.id} className="border border-blue-700 bg-gray-900 rounded-lg p-3">
                        <input
                          type="text"
                          value={draft.title || ''}
                          onChange={e => setDraft({ ...draft, title: e.target.value })}
                          placeholder="Title (optional)"
                          className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100 mb-2"
                        />
                        <textarea
                          value={draft.text}
                          onChange={e => setDraft({ ...draft, text: e.target.value })}
                          placeholder="Prompt text"
                          rows={4}
                          className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100 mb-2 font-mono"
                        />
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
                          <NumField
                            label="Width"
                            value={draft.width}
                            placeholder="job default"
                            onChange={v => setDraft({ ...draft, width: v })}
                          />
                          <NumField
                            label="Height"
                            value={draft.height}
                            placeholder="job default"
                            onChange={v => setDraft({ ...draft, height: v })}
                          />
                          <NumField
                            label="Seed"
                            help={SEED_HELP}
                            value={draft.seed}
                            placeholder="job default"
                            onChange={v => setDraft({ ...draft, seed: v })}
                          />
                          <NumField
                            label="LoRA Scale"
                            help={LORA_HELP}
                            value={draft.network_multiplier}
                            placeholder="1.0"
                            step="0.05"
                            allowDecimal
                            onChange={v => setDraft({ ...draft, network_multiplier: v })}
                          />
                        </div>
                        <div className="mb-2">
                          <div className="text-xs text-gray-400 mb-1">Tags</div>
                          <div className="flex flex-wrap gap-1 items-center">
                            {draft.tags.map(tag => (
                              <span
                                key={tag}
                                className="text-xs bg-blue-900/50 text-blue-200 px-2 py-0.5 rounded-full flex items-center gap-1"
                              >
                                {tag}
                                <button onClick={() => toggleDraftTag(tag)} className="hover:text-white">
                                  <X className="w-3 h-3" />
                                </button>
                              </span>
                            ))}
                            <input
                              type="text"
                              placeholder="add tag, Enter"
                              className="bg-gray-950 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-200"
                              onKeyDown={e => {
                                if (e.key === 'Enter') {
                                  e.preventDefault();
                                  addDraftTagFromInput((e.target as HTMLInputElement).value);
                                  (e.target as HTMLInputElement).value = '';
                                }
                              }}
                            />
                          </div>
                        </div>
                        {library.sets.length > 0 && (
                          <div className="mb-2">
                            <div className="text-xs text-gray-400 mb-1">Sets</div>
                            <div className="flex flex-wrap gap-1">
                              {library.sets.map(s => (
                                <button
                                  key={s.id}
                                  onClick={() => toggleDraftSet(s.id)}
                                  className={classNames(
                                    'text-xs px-2 py-0.5 rounded-full border',
                                    draft.setIds.includes(s.id)
                                      ? 'bg-gray-700 border-gray-500 text-white'
                                      : 'border-gray-700 text-gray-400 hover:bg-gray-800',
                                  )}
                                >
                                  <span
                                    className="inline-block w-2 h-2 rounded-full mr-1"
                                    style={{ backgroundColor: s.color }}
                                  />
                                  {s.name}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                        <div className="flex justify-end gap-2 mt-2">
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
                  return (
                    <div
                      key={p.id}
                      className="border border-gray-800 hover:border-gray-700 bg-gray-900/50 rounded-lg p-3 group"
                    >
                      <div className="flex items-start gap-2">
                        <div className="flex-1 min-w-0">
                          {p.title && (
                            <div className="text-sm font-medium text-gray-200 mb-1">{p.title}</div>
                          )}
                          <div className="text-sm text-gray-300 whitespace-pre-wrap break-words">
                            {p.text || <span className="text-gray-600 italic">empty</span>}
                          </div>
                          {(p.width || p.height || p.seed !== undefined || p.network_multiplier !== undefined) && (
                            <div className="flex flex-wrap gap-1 mt-2 text-xs text-gray-400">
                              {(p.width || p.height) && (
                                <span className="px-2 py-0.5 bg-gray-800 rounded">
                                  {p.width || '?'}×{p.height || '?'}
                                </span>
                              )}
                              {p.seed !== undefined && (
                                <span className="px-2 py-0.5 bg-gray-800 rounded">seed {p.seed}</span>
                              )}
                              {p.network_multiplier !== undefined && (
                                <span className="px-2 py-0.5 bg-gray-800 rounded">
                                  LoRA {p.network_multiplier}
                                </span>
                              )}
                            </div>
                          )}
                          {(p.tags.length > 0 || p.setIds.length > 0) && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {p.setIds.map(sid => {
                                const set = library.sets.find(s => s.id === sid);
                                if (!set) return null;
                                return (
                                  <span
                                    key={sid}
                                    className="text-xs px-2 py-0.5 rounded-full text-white"
                                    style={{ backgroundColor: set.color }}
                                  >
                                    {set.name}
                                  </span>
                                );
                              })}
                              {p.tags.map(tag => (
                                <span
                                  key={tag}
                                  className="text-xs bg-gray-800 text-gray-300 px-2 py-0.5 rounded-full"
                                >
                                  #{tag}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                        <div className="flex gap-1 opacity-0 group-hover:opacity-100">
                          <button
                            onClick={() => startEdit(p)}
                            className="p-1 text-gray-300 hover:text-white hover:bg-gray-800 rounded"
                            title="Edit"
                          >
                            <Pencil className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => deletePrompt(p.id)}
                            className="p-1 text-red-400 hover:text-red-300 hover:bg-gray-800 rounded"
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </MainContent>
      <GeneratePromptsModal
        open={generateOpen}
        onClose={() => setGenerateOpen(false)}
        onGenerate={handleGenerated}
        sets={library.sets}
      />
    </>
  );
}
