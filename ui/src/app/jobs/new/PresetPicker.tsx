'use client';

import { useEffect, useMemo, useState } from 'react';
import { Button } from '@headlessui/react';
import { LuCheck, LuCpu, LuZap } from 'react-icons/lu';
import { JobConfig } from '@/types';
import { CONFIG_PRESETS, ConfigPreset, recommendPreset } from './configPresets';
import PresetConfirmModal from './PresetConfirmModal';
import { apiClient } from '@/utils/api';

interface PresetPickerProps {
  /** modelArch.name from options.ts (e.g. "flux", "qwen_image", "wan22_14b:t2v"). */
  archName: string;
  /** Detected total VRAM in GB. Pass 0/undefined when unknown. */
  detectedVramGB?: number;
  /** GPU model name to show in the heading; falls back to "your GPU". */
  gpuName?: string;
  /** Full current config — needed to compute the diff in the confirm modal. */
  jobConfig: JobConfig;
  /** Same `(value, path)` setter the rest of the SimpleJob form uses. */
  setJobConfig: (value: any, key: string) => void;
}

export default function PresetPicker({
  archName,
  detectedVramGB,
  gpuName,
  jobConfig,
  setJobConfig,
}: PresetPickerProps) {
  const builtIn = CONFIG_PRESETS[archName] || [];
  const [userPresets, setUserPresets] = useState<ConfigPreset[]>([]);

  useEffect(() => {
    apiClient
      .get('/api/presets')
      .then(res => {
        const all: any[] = res.data?.presets || [];
        const mapped: ConfigPreset[] = all
          .filter(p => Array.isArray(p.modelArchs) && p.modelArchs.includes(archName))
          .map(p => ({
            id: `user:${p.id}`,
            label: p.name || 'Custom preset',
            description: p.description || 'User-defined preset',
            approxVramGB: typeof p.approxVramGB === 'number' ? p.approxVramGB : 0,
            tier: 'balanced',
            overrides: p.overrides || {},
          }));
        setUserPresets(mapped);
      })
      .catch(err => console.error('Failed to load user presets:', err));
  }, [archName]);

  const presets = useMemo(() => [...builtIn, ...userPresets], [builtIn, userPresets]);

  const recommended = useMemo(() => {
    if (!builtIn.length || !detectedVramGB) return null;
    return recommendPreset(builtIn, detectedVramGB);
  }, [builtIn, detectedVramGB]);

  const [pendingPreset, setPendingPreset] = useState<ConfigPreset | null>(null);

  if (presets.length === 0) {
    return null;
  }

  // Sort by approxVramGB; user presets with 0 fall to the right end.
  const ordered = [...presets].sort((a, b) => (a.approxVramGB || 999) - (b.approxVramGB || 999));

  const gpuLabel = gpuName ? gpuName.replace('NVIDIA ', '') : 'your GPU';
  const vramLabel = detectedVramGB ? `${detectedVramGB} GB VRAM` : 'unknown VRAM';

  return (
    <>
      <div className="mb-4 rounded-lg border border-gray-700 bg-gray-900/60 p-3">
        <div className="mb-2 flex items-center gap-2 text-sm text-gray-200">
          <LuCpu className="h-4 w-4 text-cyan-300" />
          <span className="font-medium">Suggested configurations</span>
          <span className="text-xs text-gray-400">
            for {gpuLabel} ({vramLabel}) — click a card to review changes before applying
          </span>
        </div>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
          {ordered.map(preset => {
            const isRecommended = recommended?.id === preset.id;
            const isCustom = preset.id.startsWith('user:');
            return (
              <Button
                key={preset.id}
                onClick={() => setPendingPreset(preset)}
                className={`flex flex-col items-start gap-1 rounded-md border px-3 py-2 text-left transition hover:bg-gray-800 ${
                  isRecommended
                    ? 'border-cyan-500 bg-cyan-900/20'
                    : isCustom
                      ? 'border-purple-600 bg-purple-900/10'
                      : 'border-gray-600 bg-transparent'
                }`}
              >
                <div className="flex w-full items-center justify-between gap-1">
                  <span className="text-sm font-medium text-gray-100">{preset.label}</span>
                  {isRecommended && (
                    <span className="flex items-center gap-1 rounded bg-cyan-700 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-white">
                      <LuCheck className="h-2.5 w-2.5" />
                      Recommended
                    </span>
                  )}
                  {isCustom && !isRecommended && (
                    <span className="rounded bg-purple-700 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-white">
                      Custom
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1 text-[11px] text-gray-400">
                  <LuZap className="h-3 w-3" />
                  {preset.approxVramGB > 0 ? `~${preset.approxVramGB} GB peak` : 'VRAM n/a'}
                </div>
                <div className="text-xs text-gray-300">{preset.description}</div>
              </Button>
            );
          })}
        </div>
        <p className="mt-2 text-[11px] text-gray-500">
          Presets are starting points — tweak any setting after applying. VRAM estimates are approximate; expect ±15%.
        </p>
      </div>
      <PresetConfirmModal
        isOpen={pendingPreset !== null}
        onClose={() => setPendingPreset(null)}
        preset={pendingPreset}
        jobConfig={jobConfig}
        setJobConfig={setJobConfig}
      />
    </>
  );
}
