import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';

const VARIABLE_PATTERN = /\$\{(\w+)=(\w[\w-]*\/)\}/g;

export interface PartialOption {
  filename: string;
  content: string;
  rawContent: string;
  source: 'default' | 'user';
  scope: string;
  variables: TemplateVariable[];
}

export interface TemplateVariable {
  name: string;
  directory: string;
  options: PartialOption[];
}

export interface CaptionPreset {
  name: string;
  rawContent: string;
  renderedContent: string;
  variables: TemplateVariable[];
  source: 'default' | 'user';
}

const DEFAULT_PRESETS_DIR = path.join(TOOLKIT_ROOT, 'caption_presets');
const USER_PRESETS_DIR = path.join(TOOLKIT_ROOT, 'data', 'caption_presets');
const DEFAULT_PARTIALS_DIR = path.join(DEFAULT_PRESETS_DIR, 'partials');
const USER_PARTIALS_DIR = path.join(USER_PRESETS_DIR, 'partials');

function scanDirForTxt(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true })
    .filter(f => f.isFile() && f.name.endsWith('.txt') && !f.name.startsWith('.'))
    .map(f => f.name);
}

/**
 * Get the 4 search directories for a given variable directory and preset name.
 * Order: default shared, default preset-specific, user shared, user preset-specific.
 * Later entries override earlier ones for same-name files.
 */
function getScopeDirs(directory: string, presetName: string): { dir: string; source: 'default' | 'user'; scope: string }[] {
  return [
    { dir: path.join(DEFAULT_PARTIALS_DIR, 'shared', directory), source: 'default', scope: 'shared' },
    { dir: path.join(DEFAULT_PARTIALS_DIR, presetName, directory), source: 'default', scope: presetName },
    { dir: path.join(USER_PARTIALS_DIR, 'shared', directory), source: 'user', scope: 'shared' },
    { dir: path.join(USER_PARTIALS_DIR, presetName, directory), source: 'user', scope: presetName },
  ];
}

/**
 * Read the raw content of a partial file (no nested variable resolution).
 * Searches the 4 scope dirs in reverse priority order.
 */
function readPartialRaw(directory: string, filename: string, presetName: string): string {
  const scopeDirs = getScopeDirs(directory, presetName);
  for (let i = scopeDirs.length - 1; i >= 0; i--) {
    const candidate = path.resolve(scopeDirs[i].dir, filename);
    const base = path.resolve(scopeDirs[i].dir) + path.sep;
    if (candidate.startsWith(base) && fs.existsSync(candidate)) {
      return fs.readFileSync(candidate, 'utf-8');
    }
  }
  return '';
}

function loadPartial(directory: string, filename: string, presetName: string, visited: Set<string> = new Set()): string {
  const key = `${directory}/${filename}`;
  if (visited.has(key)) return '';
  visited.add(key);

  const content = readPartialRaw(directory, filename, presetName);
  if (!content) return '';

  // Recursively resolve nested variables to first option
  return content.replace(VARIABLE_PATTERN, (_match, _varName, dirName) => {
    const options = getMergedPartials(dirName, presetName, 0); // depth=0 to skip nested variable discovery
    if (options.length === 0) return _match;
    return loadPartial(dirName, options[0].filename, presetName, new Set(visited));
  });
}

const MAX_VARIABLE_DEPTH = 4;

function getMergedPartials(directory: string, presetName: string, depth: number = MAX_VARIABLE_DEPTH): PartialOption[] {
  const scopeDirs = getScopeDirs(directory, presetName);
  const merged = new Map<string, { source: 'default' | 'user'; scope: string }>();

  // Process in order: later entries override earlier ones for same-name files
  for (const { dir, source, scope } of scopeDirs) {
    for (const filename of scanDirForTxt(dir)) {
      merged.set(filename, { source, scope });
    }
  }

  // Build options with raw content and nested variables
  const result: PartialOption[] = [];
  for (const [filename, { source, scope }] of merged) {
    const rawContent = readPartialRaw(directory, filename, presetName);
    const nestedVars = depth > 0 ? parseVariables(rawContent, presetName, depth - 1) : [];
    result.push({
      filename,
      rawContent,
      content: loadPartial(directory, filename, presetName),
      source,
      scope,
      variables: nestedVars,
    });
  }

  result.sort((a, b) => a.filename.localeCompare(b.filename));
  return result;
}

function parseVariables(rawContent: string, presetName: string, depth: number = MAX_VARIABLE_DEPTH): TemplateVariable[] {
  const variables: TemplateVariable[] = [];
  const seen = new Set<string>();

  let match: RegExpExecArray | null;
  const regex = new RegExp(VARIABLE_PATTERN.source, 'g');
  while ((match = regex.exec(rawContent)) !== null) {
    const varName = match[1];
    if (seen.has(varName)) continue;
    seen.add(varName);

    const directory = match[2]; // e.g. "modes/"
    const options = getMergedPartials(directory, presetName, depth);

    variables.push({ name: varName, directory, options });
  }

  return variables;
}

function renderTemplate(rawContent: string, selections: Record<string, string>): string {
  return rawContent.replace(VARIABLE_PATTERN, (_match, varName) => {
    return selections[varName] ?? '';
  });
}

export interface PartialFile {
  name: string;
  content: string;
  directory: string;
  source: 'default' | 'user';
  scope: string;
}

export async function GET(request: NextRequest) {
  // Scan both default and user preset directories, merge by name
  const defaultFiles = scanDirForTxt(DEFAULT_PRESETS_DIR);
  const userFiles = scanDirForTxt(USER_PRESETS_DIR);

  const presetMap = new Map<string, { filePath: string; source: 'default' | 'user' }>();

  for (const filename of defaultFiles) {
    presetMap.set(filename, { filePath: path.join(DEFAULT_PRESETS_DIR, filename), source: 'default' });
  }
  // User presets override defaults by name
  for (const filename of userFiles) {
    presetMap.set(filename, { filePath: path.join(USER_PRESETS_DIR, filename), source: 'user' });
  }

  const presets: CaptionPreset[] = [];
  for (const [filename, { filePath, source }] of presetMap) {
    const rawContent = fs.readFileSync(filePath, 'utf-8');
    const name = filename.replace(/\.txt$/, '');
    const variables = parseVariables(rawContent, name);

    // Render with first option of each variable as default
    const defaultSelections: Record<string, string> = {};
    for (const variable of variables) {
      if (variable.options.length > 0) {
        defaultSelections[variable.name] = variable.options[0].content;
      }
    }
    const renderedContent = renderTemplate(rawContent, defaultSelections);

    presets.push({ name, rawContent, renderedContent, variables, source });
  }

  presets.sort((a, b) => a.name.localeCompare(b.name));

  const includePartials = request.nextUrl.searchParams.get('includePartials') === 'true';

  if (includePartials) {
    // Scan all scopes (shared + preset-named dirs) from both default and user partials
    const scopes = new Set<string>();

    for (const partialsDir of [DEFAULT_PARTIALS_DIR, USER_PARTIALS_DIR]) {
      if (fs.existsSync(partialsDir)) {
        for (const entry of fs.readdirSync(partialsDir, { withFileTypes: true })) {
          if (entry.isDirectory() && !entry.name.startsWith('.')) {
            scopes.add(entry.name);
          }
        }
      }
    }

    const partials: PartialFile[] = [];
    for (const scope of scopes) {
      // Scan variable directories within this scope
      const directories = new Set<string>();

      for (const partialsDir of [DEFAULT_PARTIALS_DIR, USER_PARTIALS_DIR]) {
        const scopeDir = path.join(partialsDir, scope);
        if (fs.existsSync(scopeDir)) {
          for (const entry of fs.readdirSync(scopeDir, { withFileTypes: true })) {
            if (entry.isDirectory() && !entry.name.startsWith('.')) {
              directories.add(entry.name);
            }
          }
        }
      }

      for (const directory of directories) {
        const defaultDir = path.join(DEFAULT_PARTIALS_DIR, scope, directory);
        const userDir = path.join(USER_PARTIALS_DIR, scope, directory);
        const fileMap = new Map<string, 'default' | 'user'>();

        for (const f of scanDirForTxt(defaultDir)) {
          fileMap.set(f, 'default');
        }
        for (const f of scanDirForTxt(userDir)) {
          fileMap.set(f, 'user');
        }

        for (const [filename, source] of fileMap) {
          const dir = source === 'user' ? userDir : defaultDir;
          const content = fs.readFileSync(path.join(dir, filename), 'utf-8');
          partials.push({
            name: filename.replace(/\.txt$/, ''),
            content,
            directory,
            source,
            scope,
          });
        }
      }
    }

    partials.sort((a, b) =>
      a.scope.localeCompare(b.scope) ||
      a.directory.localeCompare(b.directory) ||
      a.name.localeCompare(b.name)
    );
    return NextResponse.json({ presets, partials });
  }

  return NextResponse.json({ presets });
}
