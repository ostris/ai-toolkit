'use client';

import { useState, useEffect, useCallback } from 'react';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { openConfirm } from '@/components/ConfirmModal';
import { Modal } from '@/components/Modal';
import { TextInput } from '@/components/formInputs';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt, FaEdit, FaChevronDown, FaChevronUp, FaCopy } from 'react-icons/fa';

interface PresetData {
  name: string;
  rawContent: string;
  variables: { name: string; options: { filename: string }[] }[];
  source: 'default' | 'user';
}

interface PartialData {
  name: string;
  content: string;
  directory: string;
  source: 'default' | 'user';
  scope: string;
}

const VARIABLE_PATTERN = /\$\{(\w+)=(\w[\w-]*\/)\}/g;

function SourceBadge({ source }: { source: 'default' | 'user' }) {
  return source === 'default' ? (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-700 text-gray-300">
      Default
    </span>
  ) : (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-900 text-blue-300">
      User
    </span>
  );
}

function ScopeBadge({ scope }: { scope: string }) {
  return scope === 'shared' ? (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-900 text-green-300">
      shared
    </span>
  ) : (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-900 text-purple-300">
      {scope}
    </span>
  );
}

export default function CaptionPresetsPage() {
  const [presets, setPresets] = useState<PresetData[]>([]);
  const [partials, setPartials] = useState<PartialData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [editName, setEditName] = useState('');
  const [editContent, setEditContent] = useState('');
  const [editDirectory, setEditDirectory] = useState(''); // empty = preset, non-empty = partial in this directory
  const [editScope, setEditScope] = useState('shared');
  const [isEditing, setIsEditing] = useState(false);
  const [isReadOnly, setIsReadOnly] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // New partial flow: scope picker then directory picker
  const [scopePickerOpen, setScopePickerOpen] = useState(false);
  const [dirPickerOpen, setDirPickerOpen] = useState(false);
  const [newDirName, setNewDirName] = useState('');
  const [pendingScope, setPendingScope] = useState('shared');

  const fetchData = useCallback(async () => {
    try {
      const res = await apiClient.get('/api/caption-presets?includePartials=true');
      setPresets(res.data.presets || []);
      setPartials(res.data.partials || []);
    } catch (error) {
      console.error('Error fetching caption presets:', error);
    }
  }, []);

  useEffect(() => {
    setIsLoading(true);
    fetchData().finally(() => setIsLoading(false));
  }, [fetchData]);

  // Get unique scopes and directories
  const scopes = [...new Set(partials.map(p => p.scope))].sort((a, b) => {
    // 'shared' always first
    if (a === 'shared') return -1;
    if (b === 'shared') return 1;
    return a.localeCompare(b);
  });

  // Also collect preset names as possible scopes for the picker
  const presetNames = presets.map(p => p.name);
  const allScopes = [...new Set([...scopes, ...presetNames])].sort((a, b) => {
    if (a === 'shared') return -1;
    if (b === 'shared') return 1;
    return a.localeCompare(b);
  });
  // Ensure 'shared' is always an option
  if (!allScopes.includes('shared')) allScopes.unshift('shared');

  const getDirectoriesForScope = (scope: string) =>
    [...new Set(partials.filter(p => p.scope === scope).map(p => p.directory))].sort();

  // All directories across all scopes (for directory picker)
  const allDirectories = [...new Set(partials.map(p => p.directory))].sort();

  const openCreate = (directory: string, scope: string) => {
    setEditName('');
    setEditContent('');
    setEditDirectory(directory);
    setEditScope(scope);
    setIsEditing(false);
    setIsReadOnly(false);
    setModalOpen(true);
  };

  const openEdit = (name: string, content: string, directory: string, scope: string, source: 'default' | 'user') => {
    setEditName(name);
    setEditContent(content);
    setEditDirectory(directory);
    setEditScope(scope);
    setIsEditing(true);
    setIsReadOnly(source === 'default');
    setModalOpen(true);
  };

  const handleCustomize = (name: string, content: string, directory: string, scope: string) => {
    // Copy a default item to user dir for editing
    setEditName(name);
    setEditContent(content);
    setEditDirectory(directory);
    setEditScope(scope);
    setIsEditing(false); // treat as new (writing to user dir)
    setIsReadOnly(false);
    setModalOpen(true);
  };

  const handleSave = async () => {
    if (!editName.trim()) return;
    setIsSaving(true);
    try {
      await apiClient.post('/api/caption-presets/save', {
        name: editName.trim(),
        content: editContent,
        directory: editDirectory || undefined,
        scope: editDirectory ? editScope : undefined,
      });
      setModalOpen(false);
      await fetchData();
    } catch (error: any) {
      console.error('Error saving preset:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = (name: string, directory: string, scope: string) => {
    const label = directory ? 'Partial' : 'Preset';
    openConfirm({
      title: `Delete ${label}`,
      message: `Are you sure you want to delete "${name}"? This cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/caption-presets/delete', {
            name,
            directory: directory || undefined,
            scope: directory ? scope : undefined,
          })
          .then(() => fetchData())
          .catch(error => console.error('Error deleting:', error));
      },
    });
  };

  const countVariables = (rawContent: string): number => {
    const seen = new Set<string>();
    let match;
    const regex = new RegExp(VARIABLE_PATTERN.source, 'g');
    while ((match = regex.exec(rawContent)) !== null) {
      seen.add(match[1]);
    }
    return seen.size;
  };

  const presetColumns: TableColumn[] = [
    {
      title: 'Name',
      key: 'name',
      render: row => (
        <button
          className="text-blue-400 hover:text-blue-300 text-sm"
          onClick={() => openEdit(row.name, row.rawContent, '', '', row.source)}
        >
          {row.name}
        </button>
      ),
    },
    {
      title: 'Source',
      key: 'source',
      className: 'w-24',
      render: row => <SourceBadge source={row.source} />,
    },
    {
      title: 'Variables',
      key: 'variables',
      className: 'w-24',
      render: row => <span className="text-gray-400 text-sm">{countVariables(row.rawContent)}</span>,
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-36 text-right',
      render: row => (
        <div className="flex gap-1 justify-end">
          {row.source === 'default' ? (
            <button
              className="text-gray-300 hover:bg-gray-600 p-2 rounded-full transition-colors"
              onClick={() => handleCustomize(row.name, row.rawContent, '', '')}
              title="Customize (copy to user dir)"
            >
              <FaCopy />
            </button>
          ) : (
            <>
              <button
                className="text-gray-300 hover:bg-gray-600 p-2 rounded-full transition-colors"
                onClick={() => openEdit(row.name, row.rawContent, '', '', row.source)}
                title="Edit"
              >
                <FaEdit />
              </button>
              <button
                className="text-gray-300 hover:bg-red-600 p-2 rounded-full transition-colors"
                onClick={() => handleDelete(row.name, '', '')}
                title="Delete"
              >
                <FaRegTrashAlt />
              </button>
            </>
          )}
        </div>
      ),
    },
  ];

  const getPartialColumns = (directory: string, scope: string): TableColumn[] => [
    {
      title: 'Name',
      key: 'name',
      render: row => (
        <button
          className="text-blue-400 hover:text-blue-300 text-sm"
          onClick={() => openEdit(row.name, row.content, directory, scope, row.source)}
        >
          {row.name}
        </button>
      ),
    },
    {
      title: 'Source',
      key: 'source',
      className: 'w-24',
      render: row => <SourceBadge source={row.source} />,
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-36 text-right',
      render: row => (
        <div className="flex gap-1 justify-end">
          {row.source === 'default' ? (
            <button
              className="text-gray-300 hover:bg-gray-600 p-2 rounded-full transition-colors"
              onClick={() => handleCustomize(row.name, row.content, directory, scope)}
              title="Customize (copy to user dir)"
            >
              <FaCopy />
            </button>
          ) : (
            <>
              <button
                className="text-gray-300 hover:bg-gray-600 p-2 rounded-full transition-colors"
                onClick={() => openEdit(row.name, row.content, directory, scope, row.source)}
                title="Edit"
              >
                <FaEdit />
              </button>
              <button
                className="text-gray-300 hover:bg-red-600 p-2 rounded-full transition-colors"
                onClick={() => handleDelete(row.name, directory, scope)}
                title="Delete"
              >
                <FaRegTrashAlt />
              </button>
            </>
          )}
        </div>
      ),
    },
  ];

  const handleNewPartialClick = () => {
    setPendingScope('shared');
    setScopePickerOpen(true);
  };

  const handleScopePickerConfirm = (scope: string) => {
    setScopePickerOpen(false);
    setPendingScope(scope);
    setNewDirName('');
    setDirPickerOpen(true);
  };

  const handleDirPickerConfirm = (dir: string) => {
    setDirPickerOpen(false);
    openCreate(dir, pendingScope);
  };

  return (
    <>
      <TopBar>
        <div className="flex items-center justify-between w-full">
          <h1 className="text-2xl font-semibold text-gray-100">Caption Presets</h1>
          <div className="flex gap-2">
            <Button
              onClick={handleNewPartialClick}
              className="text-gray-200 bg-gray-700 px-3 py-1.5 rounded-md hover:bg-gray-600 transition-colors text-sm"
            >
              New Partial
            </Button>
            <Button
              onClick={() => openCreate('', '')}
              className="text-gray-200 bg-slate-600 px-3 py-1.5 rounded-md hover:bg-slate-500 transition-colors text-sm"
            >
              New Preset
            </Button>
          </div>
        </div>
      </TopBar>

      <MainContent>
        {/* Help Section */}
        <div className="mb-6 bg-gray-900 rounded-md">
          <button
            onClick={() => setHelpOpen(v => !v)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-gray-300 uppercase tracking-wide hover:bg-gray-800 rounded-md transition-colors"
          >
            <span>How Caption Presets Work</span>
            {helpOpen ? <FaChevronUp className="text-gray-500" /> : <FaChevronDown className="text-gray-500" />}
          </button>
          {helpOpen && (
            <div className="px-4 pb-4 text-sm text-gray-400 space-y-3">
              <p>
                <strong className="text-gray-300">Presets</strong> are system prompts that tell the AI how to caption
                your images or videos. Default presets live in <code className="text-gray-300">caption_presets/</code>{' '}
                (git-tracked). User presets are saved to <code className="text-gray-300">data/caption_presets/</code>{' '}
                (gitignored).
              </p>
              <p>
                <strong className="text-gray-300">Partials</strong> are reusable text snippets organized by scope and
                variable directory under <code className="text-gray-300">caption_presets/partials/</code>.
                The <code className="text-gray-300">shared/</code> scope is inherited by all presets. Preset-specific
                scopes (e.g. <code className="text-gray-300">Z-Image/</code>) are only visible to that preset.
              </p>
              <div>
                <strong className="text-gray-300">Template Variables</strong>
                <p className="mt-1">
                  Presets can include template variables that reference a partials directory. The syntax is:
                </p>
                <code className="block bg-gray-800 rounded p-2 mt-1 text-gray-300">
                  {'${VariableName=dirname/}'}
                </code>
                <p className="mt-1">
                  All <code className="text-gray-300">.txt</code> files in that subdirectory become dropdown options,
                  merged from both shared and preset-specific scopes.
                </p>
              </div>
              <div>
                <strong className="text-gray-300">Scoped Partials</strong>
                <p className="mt-1">
                  Partials in <code className="text-gray-300">shared/</code> are available to all presets. Partials
                  under a preset name (e.g. <code className="text-gray-300">Z-Image/modes/</code>) are only
                  visible when resolving variables for that preset. This keeps preset-specific options from
                  cluttering other presets&apos; dropdowns.
                </p>
              </div>
              <div>
                <strong className="text-gray-300">User Overrides</strong>
                <p className="mt-1">
                  User partials in <code className="text-gray-300">data/caption_presets/partials/</code> are merged
                  with defaults. Same-name files in the user directory override the default. This keeps{' '}
                  <code className="text-gray-300">git status</code> clean while allowing full customization.
                </p>
              </div>
              <div>
                <strong className="text-gray-300">Example</strong>
                <pre className="bg-gray-800 rounded p-2 mt-1 text-gray-300 text-xs overflow-x-auto whitespace-pre-wrap">
{`You are an image captioning assistant.

Describe the image in detail using the following style:
\${Style=styles/}

Always be accurate and descriptive.`}
                </pre>
                <p className="mt-1">
                  This creates a &quot;Style&quot; dropdown populated from all <code className="text-gray-300">.txt</code>{' '}
                  files in the <code className="text-gray-300">partials/shared/styles/</code> and{' '}
                  <code className="text-gray-300">partials/&lt;preset-name&gt;/styles/</code> directories.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Presets Table */}
        <div className="mb-6">
          <h2 className="text-sm font-semibold text-gray-300 mb-2 uppercase tracking-wide">Presets</h2>
          <UniversalTable columns={presetColumns} rows={presets} isLoading={isLoading} onRefresh={fetchData} />
        </div>

        {/* Partials grouped by scope, then by directory */}
        {scopes.map(scope => {
          const scopeDirs = getDirectoriesForScope(scope);
          return (
            <div key={scope} className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
                  Partials
                </h2>
                <ScopeBadge scope={scope} />
              </div>
              {scopeDirs.map(directory => (
                <div key={`${scope}-${directory}`} className="mb-4 ml-2">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm text-gray-400">
                      {directory}/
                    </h3>
                    <Button
                      onClick={() => openCreate(directory, scope)}
                      className="text-gray-300 bg-gray-700 px-2 py-1 rounded hover:bg-gray-600 transition-colors text-xs"
                    >
                      + Add
                    </Button>
                  </div>
                  <UniversalTable
                    columns={getPartialColumns(directory, scope)}
                    rows={partials.filter(p => p.scope === scope && p.directory === directory)}
                    isLoading={isLoading}
                    onRefresh={fetchData}
                  />
                </div>
              ))}
            </div>
          );
        })}
      </MainContent>

      {/* Edit/Create Modal */}
      <Modal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        title={`${isReadOnly ? 'View' : isEditing ? 'Edit' : 'New'} ${editDirectory ? `Partial (${editScope}/${editDirectory})` : 'Preset'}`}
        size="lg"
      >
        <div className="space-y-4">
          <TextInput
            label="Name"
            value={editName}
            onChange={setEditName}
            disabled={isEditing}
            placeholder={editDirectory ? 'my-partial' : 'my-preset'}
          />
          <div>
            <label className="block text-xs mb-1 mt-2 text-gray-300">Content</label>
            <textarea
              value={editContent}
              onChange={e => setEditContent(e.target.value)}
              rows={14}
              readOnly={isReadOnly}
              className={`w-full text-sm px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent font-mono text-gray-200 resize-y ${isReadOnly ? 'opacity-60 cursor-not-allowed' : ''}`}
              placeholder={
                editDirectory
                  ? 'Enter partial content...'
                  : 'Enter preset system prompt...\n\nUse ${VarName=dirname/} to reference a partials directory.'
              }
            />
          </div>
          {/* Variable preview for presets */}
          {!editDirectory && countVariables(editContent) > 0 && (
            <div className="text-xs text-gray-400">
              <span className="font-semibold text-gray-300">Variables detected: </span>
              {(() => {
                const vars: string[] = [];
                let match;
                const regex = new RegExp(VARIABLE_PATTERN.source, 'g');
                while ((match = regex.exec(editContent)) !== null) {
                  if (!vars.includes(match[1])) vars.push(match[1]);
                }
                return vars.join(', ');
              })()}
            </div>
          )}
          <div className="flex justify-end gap-2 pt-2">
            <Button
              onClick={() => setModalOpen(false)}
              className="text-gray-300 bg-gray-700 px-4 py-1.5 rounded-md hover:bg-gray-600 transition-colors text-sm"
            >
              {isReadOnly ? 'Close' : 'Cancel'}
            </Button>
            {!isReadOnly && (
              <Button
                onClick={handleSave}
                disabled={isSaving || !editName.trim()}
                className="text-gray-200 bg-slate-600 px-4 py-1.5 rounded-md hover:bg-slate-500 transition-colors disabled:opacity-50 text-sm"
              >
                {isSaving ? 'Saving...' : 'Save'}
              </Button>
            )}
          </div>
        </div>
      </Modal>

      {/* Scope Picker Modal for New Partial */}
      <Modal
        isOpen={scopePickerOpen}
        onClose={() => setScopePickerOpen(false)}
        title="Choose Scope"
        size="sm"
      >
        <div className="space-y-3">
          <p className="text-sm text-gray-400">
            Select which scope this partial belongs to. &quot;shared&quot; partials are available to all presets.
            Preset-specific partials are only visible to that preset.
          </p>
          {allScopes.map(scope => (
            <button
              key={scope}
              onClick={() => handleScopePickerConfirm(scope)}
              className="w-full text-left px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm text-gray-200 transition-colors flex items-center gap-2"
            >
              <ScopeBadge scope={scope} />
              {scope}
            </button>
          ))}
        </div>
      </Modal>

      {/* Directory Picker Modal for New Partial */}
      <Modal
        isOpen={dirPickerOpen}
        onClose={() => setDirPickerOpen(false)}
        title="Choose Variable Directory"
        size="sm"
      >
        <div className="space-y-3">
          <p className="text-sm text-gray-400">
            Select which variable directory this partial belongs to, or create a new one.
          </p>
          {allDirectories.map(dir => (
            <button
              key={dir}
              onClick={() => handleDirPickerConfirm(dir)}
              className="w-full text-left px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm text-gray-200 transition-colors"
            >
              {dir}/
            </button>
          ))}
          <div className="flex gap-2 pt-2 border-t border-gray-700">
            <input
              type="text"
              value={newDirName}
              onChange={e => setNewDirName(e.target.value)}
              placeholder="new-directory-name"
              className="flex-1 bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200"
            />
            <Button
              onClick={() => {
                if (newDirName.trim()) handleDirPickerConfirm(newDirName.trim());
              }}
              disabled={!newDirName.trim()}
              className="text-gray-200 bg-slate-600 px-3 py-1.5 rounded-md hover:bg-slate-500 transition-colors disabled:opacity-50 text-sm"
            >
              Create
            </Button>
          </div>
        </div>
      </Modal>
    </>
  );
}
