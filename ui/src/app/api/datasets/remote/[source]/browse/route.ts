/**
 * GET /api/datasets/remote/[source]/browse
 *
 * Calls the Python CLI to run the registered plugin's browse() method and
 * returns normalized groups + import_fields for the generic browse modal.
 */
import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import { existsSync } from 'fs';
import { TOOLKIT_ROOT } from '@/paths';

function getPythonPath(): string {
  for (const dir of ['.venv', 'venv']) {
    const p = path.join(TOOLKIT_ROOT, dir, 'bin', 'python');
    if (existsSync(p)) return p;
    const pw = path.join(TOOLKIT_ROOT, dir, 'Scripts', 'python.exe');
    if (existsSync(pw)) return pw;
  }
  return 'python';
}

// Short-lived in-memory cache to avoid spawning a Python process on every
// modal open or React StrictMode double-invocation.
const browseCache = new Map<string, { data: unknown; expires: number }>();
const BROWSE_CACHE_TTL_MS = 30_000; // 30 seconds

export async function GET(
  _req: Request,
  { params }: { params: { source: string } },
) {
  const { source } = params;

  const cached = browseCache.get(source);
  if (cached && cached.expires > Date.now()) {
    return NextResponse.json(cached.data);
  }

  return new Promise<NextResponse>(resolve => {
    const child = spawn(
      getPythonPath(),
      ['-m', 'toolkit.dataset_sources.cli', 'browse', source],
      { cwd: TOOLKIT_ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
    );

    let out = '';
    let err = '';
    child.stdout.on('data', (d: Buffer) => (out += d.toString()));
    child.stderr.on('data', (d: Buffer) => (err += d.toString()));

    child.on('close', code => {
      if (code !== 0) {
        // CLI may write {"error": "..."} to stdout even on non-zero exit
        let pluginError: string | undefined;
        try {
          const parsed = JSON.parse(out);
          if (parsed?.error) pluginError = parsed.error;
        } catch { /* ignore */ }
        const msg = pluginError || err.trim() || 'unknown error';
        resolve(
          NextResponse.json(
            { error: `Browse failed for "${source}": ${msg}` },
            { status: 502 },
          ),
        );
        return;
      }
      try {
        const data = JSON.parse(out);
        if (data.error) {
          resolve(NextResponse.json({ error: data.error }, { status: 400 }));
        } else {
          browseCache.set(source, { data, expires: Date.now() + BROWSE_CACHE_TTL_MS });
          resolve(NextResponse.json(data));
        }
      } catch {
        resolve(NextResponse.json({ error: 'Invalid response from plugin' }, { status: 500 }));
      }
    });
  });
}
