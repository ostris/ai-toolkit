/**
 * GET /api/datasets/remote/plugins
 *
 * Returns the list of registered data-source plugins by calling the Python CLI.
 * The datasets page uses this to dynamically render "Browse <Name>" buttons.
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

export async function GET() {
  return new Promise<NextResponse>(resolve => {
    const child = spawn(
      getPythonPath(),
      ['-m', 'toolkit.dataset_sources.cli', 'plugins'],
      { cwd: TOOLKIT_ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
    );

    let out = '';
    child.stdout.on('data', (d: Buffer) => (out += d.toString()));
    child.on('close', code => {
      if (code !== 0) {
        resolve(NextResponse.json({ error: 'Failed to list plugins' }, { status: 502 }));
        return;
      }
      try {
        resolve(NextResponse.json(JSON.parse(out)));
      } catch {
        resolve(NextResponse.json({ error: 'Invalid response from CLI' }, { status: 500 }));
      }
    });
  });
}
