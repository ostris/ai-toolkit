import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { spawn, ChildProcess } from 'child_process';
import { getDatasetsRoot } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';

interface ScoringState {
  status: 'idle' | 'running' | 'completed' | 'cancelled' | 'error';
  scored: number;
  total: number;
  error?: string;
  process?: ChildProcess;
}

// In-memory store for scoring state per dataset
const scoringStates = new Map<string, ScoringState>();

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];

function findImagesRecursively(dir: string): string[] {
  let results: string[] = [];
  const items = fs.readdirSync(dir, { withFileTypes: true });
  for (const item of items) {
    if (item.isSymbolicLink()) continue;
    const itemPath = path.join(dir, item.name);
    if (item.isDirectory() && item.name !== '_controls' && !item.name.startsWith('.')) {
      results = results.concat(findImagesRecursively(itemPath));
    } else if (item.isFile()) {
      const ext = path.extname(item.name).toLowerCase();
      if (IMAGE_EXTENSIONS.includes(ext) && !item.name.startsWith('trash_')) {
        results.push(itemPath);
      }
    }
  }
  return results;
}

function getPythonPath(): string {
  const venvDirs = ['.venv', 'venv'];
  const isWindows = process.platform === 'win32';
  for (const venvDir of venvDirs) {
    const candidate = isWindows
      ? path.join(TOOLKIT_ROOT, venvDir, 'Scripts', 'python.exe')
      : path.join(TOOLKIT_ROOT, venvDir, 'bin', 'python');
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return 'python3';
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const datasetName = searchParams.get('datasetName');

  if (!datasetName || datasetName.includes('..') || datasetName.includes('/') || datasetName.includes('\\')) {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  const state = scoringStates.get(datasetName);
  if (!state) {
    return NextResponse.json({ status: 'idle', scored: 0, total: 0 });
  }

  return NextResponse.json({
    status: state.status,
    scored: state.scored,
    total: state.total,
    error: state.error,
  });
}

export async function POST(request: Request) {
  const body = await request.json();
  const { datasetName } = body;

  if (!datasetName || typeof datasetName !== 'string' || datasetName.trim() === '') {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }
  if (datasetName.includes('..') || datasetName.includes('/') || datasetName.includes('\\')) {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  const datasetsPath = await getDatasetsRoot();
  const datasetFolder = path.join(datasetsPath, datasetName);

  if (!datasetFolder.startsWith(datasetsPath)) {
    return NextResponse.json({ error: 'Invalid dataset path' }, { status: 400 });
  }

  if (!fs.existsSync(datasetFolder)) {
    return NextResponse.json({ error: `Dataset '${datasetName}' not found` }, { status: 404 });
  }

  const existing = scoringStates.get(datasetName);
  if (existing && existing.status === 'running') {
    return NextResponse.json({ error: 'Scoring already in progress' }, { status: 409 });
  }

  const images = findImagesRecursively(datasetFolder);
  const total = images.length;

  const state: ScoringState = { status: 'running', scored: 0, total };
  scoringStates.set(datasetName, state);

  const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'score_images.py');
  const pythonPath = getPythonPath();

  const proc = spawn(pythonPath, [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] });
  state.process = proc;

  // Send image list to stdin
  const input = JSON.stringify({ images });
  proc.stdin.write(input);
  proc.stdin.end();

  let stdoutBuffer = '';
  proc.stdout.on('data', (data: Buffer) => {
    stdoutBuffer += data.toString();
    const lines = stdoutBuffer.split('\n');
    stdoutBuffer = lines.pop() ?? '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('PROGRESS:')) {
        const parts = trimmed.split(':');
        if (parts.length === 3) {
          state.scored = parseInt(parts[1], 10);
          state.total = parseInt(parts[2], 10);
        }
      } else if (trimmed.startsWith('ERROR:')) {
        state.error = trimmed.slice(6);
      }
    }
  });

  proc.stderr.on('data', (data: Buffer) => {
    // Log stderr but don't fail - some Python libs output to stderr
    console.error(`[scoreImages] ${datasetName}: ${data.toString()}`);
  });

  proc.on('close', (code: number | null) => {
    const current = scoringStates.get(datasetName);
    if (current && current.status === 'running') {
      if (code === 0) {
        current.status = 'completed';
        current.scored = current.total;
      } else {
        current.status = 'error';
        if (!current.error) {
          current.error = `Process exited with code ${code}`;
        }
      }
      current.process = undefined;
    }
  });

  proc.on('error', (err: Error) => {
    const current = scoringStates.get(datasetName);
    if (current) {
      current.status = 'error';
      current.error = err.message;
      current.process = undefined;
    }
  });

  return NextResponse.json({ status: 'running', scored: 0, total });
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const datasetName = searchParams.get('datasetName');

  if (!datasetName || datasetName.includes('..') || datasetName.includes('/') || datasetName.includes('\\')) {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  const state = scoringStates.get(datasetName);
  if (!state || state.status !== 'running') {
    return NextResponse.json({ error: 'No active scoring for this dataset' }, { status: 404 });
  }

  if (state.process) {
    state.process.kill('SIGTERM');
    state.process = undefined;
  }
  state.status = 'cancelled';

  return NextResponse.json({ status: 'cancelled' });
}
