import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { spawn, ChildProcess } from 'child_process';
import { getDatasetsRoot } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';

interface CaptioningState {
  status: 'idle' | 'running' | 'completed' | 'cancelled' | 'error';
  captioned: number;
  total: number;
  error?: string;
  process?: ChildProcess;
  downloading?: boolean;
}

// In-memory store for captioning state per dataset
const captioningStates = new Map<string, CaptioningState>();

const CAPTIONABLE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm'];

const ALLOWED_MODELS = new Set([
  'Qwen/Qwen3-VL-4B-Instruct',
  'Qwen/Qwen3-VL-8B-Instruct',
  'prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1',
  'prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it',
]);

function findCaptionableFiles(dir: string): string[] {
  let results: string[] = [];
  const items = fs.readdirSync(dir, { withFileTypes: true });
  for (const item of items) {
    if (item.isSymbolicLink()) continue;
    const itemPath = path.join(dir, item.name);
    if (item.isDirectory() && item.name !== '_controls' && !item.name.startsWith('.')) {
      results = results.concat(findCaptionableFiles(itemPath));
    } else if (item.isFile()) {
      const ext = path.extname(item.name).toLowerCase();
      if (CAPTIONABLE_EXTENSIONS.includes(ext) && !item.name.startsWith('trash_')) {
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

  const state = captioningStates.get(datasetName);
  if (!state) {
    return NextResponse.json({ status: 'idle', captioned: 0, total: 0 });
  }

  return NextResponse.json({
    status: state.status,
    captioned: state.captioned,
    total: state.total,
    error: state.error,
    downloading: state.downloading,
  });
}

export async function POST(request: Request) {
  const body = await request.json();
  const { datasetName, triggerWord, systemPrompt, modelId } = body;

  if (!datasetName || typeof datasetName !== 'string' || datasetName.trim() === '') {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }
  if (datasetName.includes('..') || datasetName.includes('/') || datasetName.includes('\\')) {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  const resolvedModelId = modelId || 'Qwen/Qwen3-VL-4B-Instruct';
  if (!ALLOWED_MODELS.has(resolvedModelId)) {
    return NextResponse.json({ error: 'Invalid model ID' }, { status: 400 });
  }

  const datasetsPath = await getDatasetsRoot();
  const datasetFolder = path.join(datasetsPath, datasetName);

  if (!datasetFolder.startsWith(datasetsPath)) {
    return NextResponse.json({ error: 'Invalid dataset path' }, { status: 400 });
  }

  if (!fs.existsSync(datasetFolder)) {
    return NextResponse.json({ error: `Dataset '${datasetName}' not found` }, { status: 404 });
  }

  const existing = captioningStates.get(datasetName);
  if (existing && existing.status === 'running') {
    return NextResponse.json({ error: 'Captioning already in progress' }, { status: 409 });
  }

  const allImages = findCaptionableFiles(datasetFolder);
  const images = allImages.filter(imgPath => {
    const base = imgPath.slice(0, imgPath.lastIndexOf('.'));
    const txtPath = base + '.txt';
    if (!fs.existsSync(txtPath)) return true;
    return fs.readFileSync(txtPath, 'utf-8').trim().length === 0;
  });
  const total = images.length;

  const state: CaptioningState = { status: 'running', captioned: 0, total };
  captioningStates.set(datasetName, state);

  const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'caption_images.py');
  const pythonPath = getPythonPath();

  const proc = spawn(pythonPath, [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] });
  state.process = proc;

  const input = JSON.stringify({
    images,
    trigger_word: (triggerWord || '').toString(),
    system_prompt: (systemPrompt || '').toString(),
    model_id: resolvedModelId,
  });
  proc.stdin.write(input);
  proc.stdin.end();

  let stdoutBuffer = '';
  proc.stdout.on('data', (data: Buffer) => {
    stdoutBuffer += data.toString();
    const lines = stdoutBuffer.split('\n');
    stdoutBuffer = lines.pop() ?? '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed === 'STATUS:downloading') {
        state.downloading = true;
      } else if (trimmed.startsWith('PROGRESS:')) {
        state.downloading = false;
        const parts = trimmed.split(':');
        if (parts.length === 3) {
          state.captioned = parseInt(parts[1], 10);
          state.total = parseInt(parts[2], 10);
        }
      } else if (trimmed.startsWith('ERROR:')) {
        state.error = trimmed.slice(6);
      }
    }
  });

  proc.stderr.on('data', (data: Buffer) => {
    // Log stderr but don't fail - Python libs often write to stderr
    console.error(`[captionImages] ${datasetName}: ${data.toString()}`);
  });

  proc.on('close', (code: number | null) => {
    const current = captioningStates.get(datasetName);
    if (current && current.status === 'running') {
      if (code === 0) {
        current.status = 'completed';
        current.captioned = current.total;
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
    const current = captioningStates.get(datasetName);
    if (current) {
      current.status = 'error';
      current.error = err.message;
      current.process = undefined;
    }
  });

  return NextResponse.json({
    status: 'running',
    captioned: 0,
    total,
  });
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const datasetName = searchParams.get('datasetName');

  if (!datasetName || datasetName.includes('..') || datasetName.includes('/') || datasetName.includes('\\')) {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  const state = captioningStates.get(datasetName);
  if (!state || state.status !== 'running') {
    return NextResponse.json({ error: 'No active captioning for this dataset' }, { status: 404 });
  }

  if (state.process) {
    state.process.kill('SIGTERM');
    state.process = undefined;
  }
  state.status = 'cancelled';

  return NextResponse.json({ status: 'cancelled' });
}
