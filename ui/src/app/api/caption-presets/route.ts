import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';

const VARIABLE_PATTERN = /\$\{(\w+)=\[([^\]]+)\]\}/g;

export interface PartialOption {
  filename: string;
  content: string;
}

export interface TemplateVariable {
  name: string;
  options: PartialOption[];
}

export interface CaptionPreset {
  name: string;
  rawContent: string;
  renderedContent: string;
  variables: TemplateVariable[];
}

function loadPartial(partialsDir: string, filename: string, visited: Set<string> = new Set()): string {
  if (visited.has(filename)) {
    return '';
  }
  visited.add(filename);

  const filePath = path.resolve(partialsDir, filename);
  // Prevent path traversal: ensure resolved path is strictly within the partials directory
  if (!filePath.startsWith(path.resolve(partialsDir) + path.sep)) {
    return '';
  }
  if (!fs.existsSync(filePath)) {
    return '';
  }

  let content = fs.readFileSync(filePath, 'utf-8');

  // Recursively resolve nested partials
  content = content.replace(VARIABLE_PATTERN, (_match, _varName, optionsStr) => {
    const filenames = optionsStr
      .split(',')
      .map((s: string) => s.trim().replace(/^["']|["']$/g, ''));
    if (filenames.length === 0) return _match;
    return loadPartial(partialsDir, filenames[0], new Set(visited));
  });

  return content;
}

function parseVariables(rawContent: string, partialsDir: string): TemplateVariable[] {
  const variables: TemplateVariable[] = [];
  const seen = new Set<string>();

  let match: RegExpExecArray | null;
  const regex = new RegExp(VARIABLE_PATTERN.source, 'g');
  while ((match = regex.exec(rawContent)) !== null) {
    const varName = match[1];
    if (seen.has(varName)) continue;
    seen.add(varName);

    const filenames = match[2]
      .split(',')
      .map((s: string) => s.trim().replace(/^["']|["']$/g, ''));

    const options: PartialOption[] = filenames.map(filename => ({
      filename,
      content: loadPartial(partialsDir, filename),
    }));

    variables.push({ name: varName, options });
  }

  return variables;
}

function renderTemplate(rawContent: string, selections: Record<string, string>): string {
  return rawContent.replace(VARIABLE_PATTERN, (_match, varName) => {
    return selections[varName] ?? '';
  });
}

export async function GET() {
  const presetsDir = path.join(TOOLKIT_ROOT, 'caption_presets');
  const partialsDir = path.join(presetsDir, 'partials');

  if (!fs.existsSync(presetsDir)) {
    return NextResponse.json({ presets: [] });
  }

  const files = fs.readdirSync(presetsDir, { withFileTypes: true });
  const presets: CaptionPreset[] = files
    .filter(f => f.isFile() && f.name.endsWith('.txt') && !f.name.startsWith('.'))
    .map(f => {
      const filePath = path.join(presetsDir, f.name);
      const rawContent = fs.readFileSync(filePath, 'utf-8');
      const name = f.name.replace(/\.txt$/, '');

      const variables = parseVariables(rawContent, partialsDir);

      // Render with first option of each variable as default
      const defaultSelections: Record<string, string> = {};
      for (const variable of variables) {
        if (variable.options.length > 0) {
          defaultSelections[variable.name] = variable.options[0].content;
        }
      }
      const renderedContent = renderTemplate(rawContent, defaultSelections);

      return { name, rawContent, renderedContent, variables };
    })
    .sort((a, b) => a.name.localeCompare(b.name));

  return NextResponse.json({ presets });
}
