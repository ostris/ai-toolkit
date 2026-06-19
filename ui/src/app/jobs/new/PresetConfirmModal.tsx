'use client';

import { useMemo, useState, useEffect } from 'react';
import { Button } from '@headlessui/react';
import { Modal } from '@/components/Modal';
import { JobConfig } from '@/types';
import { ConfigPreset } from './configPresets';

interface PresetConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  preset: ConfigPreset | null;
  jobConfig: JobConfig;
  /** Same setter shape SimpleJob hands around. */
  setJobConfig: (value: any, key: string) => void;
}

/**
 * Read a value from a deeply-nested object using the same path syntax
 * `setNestedValue` accepts (e.g. "config.process[0].model.quantize").
 * Returns undefined when any segment along the path is missing.
 */
function getNestedValue(obj: unknown, path: string): unknown {
  const pathArray: (string | number)[] = [];
  const re = /([^[.\]]+)|\[(\d+)\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(path)) !== null) {
    if (m[1] !== undefined) pathArray.push(m[1]);
    else pathArray.push(Number(m[2]));
  }
  let current: any = obj;
  for (const key of pathArray) {
    if (current == null) return undefined;
    current = current[key as any];
  }
  return current;
}

// Friendly labels for common config keys. Fallback rule below.
const LABEL_MAP: Record<string, string> = {
  'config.process[0].model.quantize': 'Quantize transformer',
  'config.process[0].model.qtype': 'Transformer quant type',
  'config.process[0].model.quantize_te': 'Quantize text encoder',
  'config.process[0].model.qtype_te': 'Text encoder quant type',
  'config.process[0].model.low_vram': 'Low VRAM mode',
  'config.process[0].model.layer_offloading': 'Layer offloading',
  'config.process[0].model.layer_offloading_transformer_percent': 'Transformer offload %',
  'config.process[0].model.layer_offloading_text_encoder_percent': 'Text encoder offload %',
  'config.process[0].train.gradient_checkpointing': 'Gradient checkpointing',
  'config.process[0].train.gradient_accumulation': 'Gradient accumulation',
  'config.process[0].train.gradient_accumulation_steps': 'Gradient accumulation steps',
  'config.process[0].train.batch_size': 'Batch size',
  'config.process[0].train.cache_text_embeddings': 'Cache text embeddings',
  'config.process[0].datasets[0].cache_latents_to_disk': 'Cache latents to disk',
  'config.process[0].datasets[0].resolution': 'Resolution buckets',
  'config.process[0].datasets[0].num_frames': 'Number of frames',
};

function friendlyLabel(path: string): string {
  if (LABEL_MAP[path]) return LABEL_MAP[path];
  // Fallback: humanize the last meaningful segment.
  const segments = path.split(/[.[\]]/).filter(Boolean);
  const tail = segments[segments.length - 1] || path;
  return tail.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatValue(value: unknown): string {
  if (value === undefined) return '(not set)';
  if (value === null) return 'null';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (Array.isArray(value)) return `[${value.join(', ')}]`;
  if (typeof value === 'object') return JSON.stringify(value);
  if (typeof value === 'string') {
    // Truncate long ARA hashes so the table stays readable
    if (value.length > 50) return value.slice(0, 47) + '…';
    return value;
  }
  return String(value);
}

function valuesEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((v, i) => valuesEqual(v, b[i]));
  }
  return false;
}

export default function PresetConfirmModal({
  isOpen,
  onClose,
  preset,
  jobConfig,
  setJobConfig,
}: PresetConfirmModalProps) {
  // Build the diff: only paths where current !== suggested.
  const rows = useMemo(() => {
    if (!preset) return [];
    return Object.entries(preset.overrides)
      .map(([path, suggested]) => ({
        path,
        label: friendlyLabel(path),
        current: getNestedValue(jobConfig, path),
        suggested,
      }))
      .filter(row => !valuesEqual(row.current, row.suggested));
  }, [preset, jobConfig]);

  // Per-row enable state. All on by default; reset whenever the preset changes.
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (preset) {
      const initial: Record<string, boolean> = {};
      for (const row of rows) initial[row.path] = true;
      setEnabled(initial);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset?.id]);

  if (!preset) return null;

  const enabledCount = rows.filter(r => enabled[r.path]).length;
  const allEnabled = rows.length > 0 && enabledCount === rows.length;

  const toggleRow = (path: string) =>
    setEnabled(prev => ({ ...prev, [path]: !prev[path] }));

  const toggleAll = () => {
    const next: Record<string, boolean> = {};
    for (const row of rows) next[row.path] = !allEnabled;
    setEnabled(next);
  };

  const handleApply = () => {
    for (const row of rows) {
      if (enabled[row.path]) {
        setJobConfig(row.suggested, row.path);
      }
    }
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`Apply preset: ${preset.label}`} size="lg">
      <div className="flex flex-col gap-3 text-sm text-gray-200">
        {rows.length === 0 ? (
          <p className="rounded-md border border-gray-700 bg-gray-900/60 px-3 py-2 text-gray-300">
            All settings already match this preset — nothing to apply.
          </p>
        ) : (
          <>
            <p className="text-xs text-gray-400">
              {rows.length} setting{rows.length === 1 ? '' : 's'} would change. Uncheck any rows you&apos;d like to
              keep at the current value.
            </p>
            <div className="max-h-[60vh] overflow-y-auto rounded-md border border-gray-700">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-gray-900 text-[11px] uppercase tracking-wide text-gray-400">
                  <tr>
                    <th className="px-2 py-2 text-left">
                      <input
                        type="checkbox"
                        checked={allEnabled}
                        ref={el => {
                          if (el) el.indeterminate = enabledCount > 0 && enabledCount < rows.length;
                        }}
                        onChange={toggleAll}
                        aria-label="Toggle all"
                      />
                    </th>
                    <th className="px-2 py-2 text-left">Setting</th>
                    <th className="px-2 py-2 text-left">Current</th>
                    <th className="px-2 py-2 text-left">→ Suggested</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(row => (
                    <tr
                      key={row.path}
                      className={`border-t border-gray-800 ${
                        enabled[row.path] ? 'bg-transparent' : 'bg-gray-950/40 opacity-60'
                      }`}
                    >
                      <td className="px-2 py-1.5 align-top">
                        <input
                          type="checkbox"
                          checked={!!enabled[row.path]}
                          onChange={() => toggleRow(row.path)}
                          aria-label={`Apply change to ${row.label}`}
                        />
                      </td>
                      <td className="px-2 py-1.5 align-top">
                        <div className="font-medium text-gray-100">{row.label}</div>
                        <div className="font-mono text-[10px] text-gray-500" title={row.path}>
                          {row.path.replace('config.process[0].', '')}
                        </div>
                      </td>
                      <td className="px-2 py-1.5 align-top font-mono text-amber-200" title={formatValue(row.current)}>
                        {formatValue(row.current)}
                      </td>
                      <td className="px-2 py-1.5 align-top font-mono text-cyan-200" title={formatValue(row.suggested)}>
                        {formatValue(row.suggested)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
        <div className="mt-2 flex items-center justify-between gap-2">
          <span className="text-xs text-gray-500">
            Preset target: ~{preset.approxVramGB} GB peak. Tweak any field after applying.
          </span>
          <div className="flex gap-2">
            <Button
              className="rounded-md border border-gray-600 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-800"
              onClick={onClose}
            >
              Cancel
            </Button>
            <Button
              className="rounded-md bg-cyan-700 px-3 py-1.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleApply}
              disabled={enabledCount === 0}
            >
              Apply {enabledCount} change{enabledCount === 1 ? '' : 's'}
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  );
}
