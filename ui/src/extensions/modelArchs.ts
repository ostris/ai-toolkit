'use client';
// Runtime-loaded model arch list. Extension packages ship a ui.tsx exporting
// AI_TOOLKIT_UI_MODELS (see README.md here); /api/model_archs transpiles them
// and this hook evaluates them in the browser with a fixed set of importable
// modules. Later modules override earlier ones by arch name, so a package
// under extensions/ can replace a built-in entry.
import React, { useEffect, useMemo } from 'react';
import * as ReactJsxRuntime from 'react/jsx-runtime';
import * as ReactJsxDevRuntime from 'react/jsx-dev-runtime';
import Link from 'next/link';
import { createGlobalState } from 'react-global-hooks';
import * as defaultSamples from '@/helpers/defaultSamples';
import * as formInputs from '@/components/formInputs';
import { apiClient } from '@/utils/api';
import * as options from '@/app/jobs/new/options';
import { ModelArch } from '@/app/jobs/new/options';
import { GroupedSelectOption } from '@/types';

// what `import ... from '<name>'` resolves to inside an extension ui module
const SHIM: { [name: string]: any } = {
  react: React,
  'react/jsx-runtime': ReactJsxRuntime,
  'react/jsx-dev-runtime': ReactJsxDevRuntime,
  'next/link': { __esModule: true, default: Link },
  '@/helpers/defaultSamples': defaultSamples,
  '@/components/formInputs': formInputs,
  '@/types': {}, // types only
  '@/app/jobs/new/options': options,
};

interface UiModule {
  id: string;
  file: string;
  mtimeMs: number;
  code: string;
}

interface ModelArchsState {
  archs: ModelArch[];
  errors: string[];
  isLoaded: boolean;
}

const state = createGlobalState<ModelArchsState>({ archs: [], errors: [], isLoaded: false });
let inflight: Promise<void> | null = null;

const evaluateModule = (mod: UiModule): ModelArch[] => {
  const module = { exports: {} as any };
  const require = (name: string) => {
    if (name in SHIM) return SHIM[name];
    throw new Error(`cannot import "${name}" (available: ${Object.keys(SHIM).join(', ')})`);
  };
  new Function('require', 'module', 'exports', mod.code)(require, module, module.exports);
  const list = module.exports.AI_TOOLKIT_UI_MODELS;
  if (!Array.isArray(list)) throw new Error('must export an AI_TOOLKIT_UI_MODELS array');
  list.forEach((a, i) => {
    if (!a || typeof a.name !== 'string' || typeof a.label !== 'string' || typeof a.group !== 'string') {
      throw new Error(`AI_TOOLKIT_UI_MODELS[${i}] needs string name, label and group`);
    }
  });
  return list as ModelArch[];
};

export const loadModelArchs = async (): Promise<void> => {
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const res = await apiClient.get('/api/model_archs');
      const modules: UiModule[] = res.data?.modules || [];
      const errors: string[] = (res.data?.errors || []).map((e: any) => `${e.id}: ${e.error}`);
      const byName = new Map<string, ModelArch>();
      for (const mod of modules) {
        try {
          for (const arch of evaluateModule(mod)) byName.set(arch.name, arch);
        } catch (e: any) {
          errors.push(`${mod.id}: ${e?.message || e}`);
        }
      }
      for (const err of errors) console.error('[model_archs]', err);
      const archs = [...byName.values()].sort((a, b) =>
        a.label.localeCompare(b.label, undefined, { sensitivity: 'base' }),
      );
      state.set({ archs, errors, isLoaded: true });
    } catch (e: any) {
      const msg = e?.response?.data?.error || e?.message || String(e);
      console.error('[model_archs]', msg);
      state.set({ ...state.get(), errors: [msg], isLoaded: true });
    } finally {
      inflight = null;
    }
  })();
  return inflight;
};

/** Model archs for the training form and Generate page. Refetches on every
 * mount (server side is mtime-cached), so navigating to a page picks up new
 * extension files; previous data stays until the reload lands. */
export const useModelArchs = (): ModelArchsState & {
  groupedModelOptions: GroupedSelectOption[];
  refresh: () => Promise<void>;
} => {
  const [current] = state.use();
  useEffect(() => {
    loadModelArchs();
  }, []);
  const groupedModelOptions = useMemo(() => options.groupModelOptions(current.archs), [current.archs]);
  return { ...current, groupedModelOptions, refresh: loadModelArchs };
};
