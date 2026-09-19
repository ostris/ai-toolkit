// Discovers and transpiles extension UI modules at request time.
//
// Every package under extensions/ and extensions_built_in/ may ship a
// `ui.tsx` (or .ts/.jsx/.js) exporting AI_TOOLKIT_UI_MODELS, the UI
// counterpart of the python AI_TOOLKIT_MODELS list. The files are read from
// disk on each call (mtime-cached), so a new or edited model shows up on the
// next page load without a rebuild. Transpiled with the SWC that ships inside
// Next (native binding, all platforms); evaluation happens in the browser,
// see src/extensions/modelArchs.ts.
import fs from 'fs';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';

const EXTENSION_DIRS = ['extensions_built_in', 'extensions'];
const UI_FILE_NAMES = ['ui.tsx', 'ui.ts', 'ui.jsx', 'ui.js'];

export interface ExtensionUiModule {
  id: string; // "<dir>/<package>"
  file: string; // absolute path
  mtimeMs: number;
  code: string; // CommonJS source
}

export interface ExtensionUiError {
  id: string;
  file: string;
  error: string;
}

const cache = new Map<string, ExtensionUiModule>();

// the swc binding is a native module resolved at runtime, outside the bundler
// graph. webpack rewrites a statically imported createRequire whose base path
// it cannot evaluate into `(void 0)(...)`, so the builtin is loaded through an
// import both bundlers are told to leave alone.
let swcPromise: Promise<any> | null = null;
const getSwc = () => {
  if (!swcPromise) {
    swcPromise = (async () => {
      const nodeModule = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ 'module');
      const req = nodeModule.createRequire(path.join(TOOLKIT_ROOT, 'ui', 'package.json'));
      return req('next/dist/build/swc');
    })();
  }
  return swcPromise;
};

const transpile = async (file: string, source: string): Promise<string> => {
  const swc = await getSwc();
  const ts = /\.tsx?$/.test(file);
  const out = await swc.transform(source, {
    filename: file,
    jsc: {
      parser: ts ? { syntax: 'typescript', tsx: file.endsWith('x') } : { syntax: 'ecmascript', jsx: true },
      transform: { react: { runtime: 'automatic', development: false } },
      target: 'es2020',
    },
    module: { type: 'commonjs' },
    sourceMaps: false,
    isModule: true,
  });
  return out.code as string;
};

const findUiFile = (pkgDir: string): string | null => {
  for (const name of UI_FILE_NAMES) {
    const file = path.join(pkgDir, name);
    try {
      if (fs.statSync(file).isFile()) return file;
    } catch {
      /* not present */
    }
  }
  return null;
};

/** All extension UI modules, built-ins first, packages sorted by name. */
export const listExtensionUiModules = async (): Promise<{
  modules: ExtensionUiModule[];
  errors: ExtensionUiError[];
}> => {
  const modules: ExtensionUiModule[] = [];
  const errors: ExtensionUiError[] = [];
  const seen = new Set<string>();
  for (const dir of EXTENSION_DIRS) {
    const root = path.join(TOOLKIT_ROOT, dir);
    let entries: fs.Dirent[] = [];
    try {
      entries = fs.readdirSync(root, { withFileTypes: true });
    } catch {
      continue;
    }
    const pkgs = entries
      .filter(e => e.isDirectory() && !e.name.startsWith('.') && !e.name.startsWith('_'))
      .map(e => e.name)
      .sort();
    for (const pkg of pkgs) {
      const file = findUiFile(path.join(root, pkg));
      if (!file) continue;
      const id = `${dir}/${pkg}`;
      seen.add(id);
      try {
        const mtimeMs = fs.statSync(file).mtimeMs;
        const cached = cache.get(id);
        if (cached && cached.file === file && cached.mtimeMs === mtimeMs) {
          modules.push(cached);
          continue;
        }
        const source = await fs.promises.readFile(file, 'utf8');
        const code = await transpile(file, source);
        const mod = { id, file, mtimeMs, code };
        cache.set(id, mod);
        modules.push(mod);
      } catch (e: any) {
        cache.delete(id);
        errors.push({ id, file, error: e?.message || String(e) });
      }
    }
  }
  for (const id of cache.keys()) if (!seen.has(id)) cache.delete(id);
  return { modules, errors };
};
