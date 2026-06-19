'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { apiClient } from '@/utils/api';
import classNames from 'classnames';
import { X, Search, Folder, Tag } from 'lucide-react';

interface PromptItem {
  id: string;
  text: string;
  title?: string;
  tags: string[];
  setIds: string[];
  width?: number;
  height?: number;
  seed?: number;
  network_multiplier?: number;
}

interface PromptSet {
  id: string;
  name: string;
  color?: string;
}

interface Library {
  prompts: PromptItem[];
  sets: PromptSet[];
}

interface Props {
  open: boolean;
  onClose: () => void;
  onConfirm: (selectedPrompts: PromptItem[]) => void;
}

export default function PromptLibraryModal({ open, onClose, onConfirm }: Props) {
  const [library, setLibrary] = useState<Library>({ prompts: [], sets: [] });
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState('');
  const [activeSet, setActiveSet] = useState<string | 'all'>('all');
  const [activeTag, setActiveTag] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setSelected(new Set());
    apiClient
      .get('/api/prompts')
      .then(res => {
        const data = res.data || {};
        setLibrary({
          prompts: (data.prompts || []) as PromptItem[],
          sets: (data.sets || []) as PromptSet[],
        });
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load prompts:', err);
        setLoading(false);
      });
  }, [open]);

  const allTags = useMemo(() => {
    const t = new Set<string>();
    library.prompts.forEach(p => p.tags?.forEach(x => t.add(x)));
    return Array.from(t).sort();
  }, [library.prompts]);

  const filtered = useMemo(() => {
    let rows = library.prompts;
    if (activeSet !== 'all') rows = rows.filter(p => p.setIds?.includes(activeSet));
    if (activeTag) rows = rows.filter(p => p.tags?.includes(activeTag));
    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter(
        p =>
          p.text?.toLowerCase().includes(q) ||
          (p.title || '').toLowerCase().includes(q) ||
          p.tags?.some(t => t.toLowerCase().includes(q)),
      );
    }
    return rows;
  }, [library.prompts, activeSet, activeTag, search]);

  if (!open) return null;

  const toggle = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAllFiltered = () => {
    setSelected(prev => {
      const next = new Set(prev);
      filtered.forEach(p => next.add(p.id));
      return next;
    });
  };

  const clearSelection = () => setSelected(new Set());

  const handleConfirm = () => {
    const items = library.prompts.filter(p => selected.has(p.id));
    onConfirm(items);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg w-[95vw] max-w-5xl h-[85vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center px-4 py-3 border-b border-gray-800">
          <div className="text-lg flex-1">Prompt Library</div>
          <Link
            href="/prompts"
            className="text-xs text-blue-400 hover:underline mr-4"
            onClick={onClose}
          >
            Open Prompt Builder →
          </Link>
          <button className="text-gray-300 hover:text-white" onClick={onClose}>
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 flex overflow-hidden">
          <div className="w-56 border-r border-gray-800 p-3 overflow-auto">
            <div className="text-xs uppercase text-gray-400 mb-2 flex items-center gap-1">
              <Folder className="w-3 h-3" /> Sets
            </div>
            <button
              onClick={() => setActiveSet('all')}
              className={classNames('w-full text-left px-2 py-1 rounded text-sm mb-1', {
                'bg-blue-900/40 text-blue-200': activeSet === 'all',
                'text-gray-300 hover:bg-gray-800': activeSet !== 'all',
              })}
            >
              All ({library.prompts.length})
            </button>
            {library.sets.map(s => {
              const count = library.prompts.filter(p => p.setIds?.includes(s.id)).length;
              return (
                <button
                  key={s.id}
                  onClick={() => setActiveSet(s.id)}
                  className={classNames(
                    'w-full text-left px-2 py-1 rounded text-sm mb-1 flex items-center gap-2',
                    activeSet === s.id ? 'bg-blue-900/40 text-blue-200' : 'text-gray-300 hover:bg-gray-800',
                  )}
                >
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color || '#888' }} />
                  <span className="truncate flex-1">{s.name}</span>
                  <span className="text-gray-500 text-xs">{count}</span>
                </button>
              );
            })}
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

          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="p-3 border-b border-gray-800 flex items-center gap-2">
              <div className="relative flex-1">
                <Search className="w-4 h-4 absolute left-2 top-2 text-gray-500" />
                <input
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Search prompts..."
                  className="w-full bg-gray-950 border border-gray-700 rounded pl-8 pr-3 py-1.5 text-sm text-gray-200"
                />
              </div>
              <button
                onClick={selectAllFiltered}
                className="text-xs text-gray-300 bg-gray-800 hover:bg-gray-700 px-2 py-1 rounded"
              >
                Select all ({filtered.length})
              </button>
              <button
                onClick={clearSelection}
                className="text-xs text-gray-300 bg-gray-800 hover:bg-gray-700 px-2 py-1 rounded"
              >
                Clear
              </button>
            </div>

            <div className="flex-1 overflow-auto p-3 space-y-2">
              {loading ? (
                <div className="text-gray-400">Loading...</div>
              ) : filtered.length === 0 ? (
                <div className="text-center text-gray-400 p-8 border border-dashed border-gray-700 rounded-lg">
                  No prompts found. Add some in the{' '}
                  <Link href="/prompts" className="text-blue-400 hover:underline" onClick={onClose}>
                    Prompt Builder
                  </Link>
                  .
                </div>
              ) : (
                filtered.map(p => {
                  const isSel = selected.has(p.id);
                  return (
                    <div
                      key={p.id}
                      onClick={() => toggle(p.id)}
                      className={classNames(
                        'border rounded-lg p-3 cursor-pointer transition-colors',
                        isSel
                          ? 'border-blue-600 bg-blue-950/30'
                          : 'border-gray-800 hover:border-gray-700 bg-gray-900/50',
                      )}
                    >
                      <div className="flex items-start gap-2">
                        <input type="checkbox" checked={isSel} readOnly className="mt-1" />
                        <div className="flex-1 min-w-0">
                          {p.title && (
                            <div className="text-sm font-medium text-gray-200 mb-1">{p.title}</div>
                          )}
                          <div className="text-sm text-gray-300 whitespace-pre-wrap break-words">
                            {p.text || <span className="text-gray-600 italic">empty</span>}
                          </div>
                          {(p.tags?.length > 0 || p.setIds?.length > 0) && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {p.setIds?.map(sid => {
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
                              {p.tags?.map(tag => (
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
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

        <div className="px-4 py-3 border-t border-gray-800 flex items-center">
          <div className="text-sm text-gray-400">{selected.size} selected</div>
          <div className="flex-1"></div>
          <button
            onClick={onClose}
            className="px-3 py-1 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded mr-2"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={selected.size === 0}
            className="px-3 py-1 text-sm text-white bg-green-600 hover:bg-green-700 rounded disabled:opacity-40"
          >
            Add {selected.size > 0 ? `${selected.size} ` : ''}to Job
          </button>
        </div>
      </div>
    </div>
  );
}
