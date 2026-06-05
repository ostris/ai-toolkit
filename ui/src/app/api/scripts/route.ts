import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT } from '@/paths';
import { resolvePythonPath } from '../../../../cron/pythonPath';

// Long-running scripts: allow up to 20 minutes.
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 1200;

const TIMEOUT_MS = 20 * 60 * 1000;
const UI_SCRIPTS_ROOT = path.join(TOOLKIT_ROOT, 'ui_scripts');
// Only allow flat script names (no path separators, no traversal).
const SCRIPT_NAME_RE = /^[A-Za-z0-9_][A-Za-z0-9_.-]*\.py$/;

const resolveScriptPath = (rawName: unknown): string | null => {
  if (typeof rawName !== 'string') return null;
  const name = rawName.trim();
  if (!SCRIPT_NAME_RE.test(name)) return null;

  const target = path.resolve(UI_SCRIPTS_ROOT, name);
  const rootWithSep = UI_SCRIPTS_ROOT.endsWith(path.sep) ? UI_SCRIPTS_ROOT : UI_SCRIPTS_ROOT + path.sep;
  if (!target.startsWith(rootWithSep)) return null;
  if (!fs.existsSync(target) || !fs.statSync(target).isFile()) return null;
  return target;
};

// Args may be a positional list or an object that becomes --key value pairs.
// Every value is stringified before being passed to spawn (no shell).
const normalizeArgs = (raw: unknown): string[] | { error: string } => {
  if (raw == null) return [];
  if (Array.isArray(raw)) {
    const out: string[] = [];
    for (const v of raw) {
      if (v == null) continue;
      if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
        out.push(String(v));
      } else {
        return { error: 'args entries must be string|number|boolean' };
      }
    }
    return out;
  }
  if (typeof raw === 'object') {
    const out: string[] = [];
    for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
      if (!/^[A-Za-z0-9_-]+$/.test(key)) return { error: `invalid arg key: ${key}` };
      const flag = `--${key}`;
      if (value === true) {
        out.push(flag);
      } else if (value === false || value == null) {
        continue;
      } else if (typeof value === 'string' || typeof value === 'number') {
        out.push(flag, String(value));
      } else {
        return { error: `args.${key} must be string|number|boolean` };
      }
    }
    return out;
  }
  return { error: 'args must be an array or object' };
};

interface RunResult {
  ok: boolean;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
  result: unknown;
  timedOut: boolean;
  error?: string;
}

// Parses the last line of stdout as JSON if possible — scripts can use this
// to return structured data alongside their human-readable logs.
const parseResult = (stdout: string): unknown => {
  const lines = stdout.trimEnd().split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (!line) continue;
    if (line.startsWith('{') || line.startsWith('[')) {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    }
    return null;
  }
  return null;
};

const runBuffered = (scriptPath: string, args: string[]): Promise<RunResult> => {
  return new Promise(resolve => {
    const child = spawn(resolvePythonPath(), ['-u', scriptPath, ...args], {
      cwd: TOOLKIT_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, TIMEOUT_MS);

    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf-8');
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf-8');
    });

    child.on('error', err => {
      clearTimeout(timer);
      resolve({
        ok: false,
        exitCode: null,
        signal: null,
        stdout,
        stderr,
        result: null,
        timedOut,
        error: err.message,
      });
    });

    child.on('close', (code, signal) => {
      clearTimeout(timer);
      resolve({
        ok: !timedOut && code === 0,
        exitCode: code,
        signal,
        stdout,
        stderr,
        result: parseResult(stdout),
        timedOut,
        error: timedOut ? 'Script timed out after 20 minutes' : undefined,
      });
    });
  });
};

// NDJSON stream: one JSON object per line so clients can parse incrementally.
const runStreaming = (scriptPath: string, args: string[]): Response => {
  const child = spawn(resolvePythonPath(), ['-u', scriptPath, ...args], {
    cwd: TOOLKIT_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
    windowsHide: true,
  });

  const encoder = new TextEncoder();
  let stdoutBuf = '';
  let stderrBuf = '';
  let timedOut = false;

  const stream = new ReadableStream({
    start(controller) {
      const send = (obj: unknown) => {
        controller.enqueue(encoder.encode(JSON.stringify(obj) + '\n'));
      };

      const timer = setTimeout(() => {
        timedOut = true;
        send({ type: 'error', message: 'Script timed out after 20 minutes' });
        child.kill('SIGKILL');
      }, TIMEOUT_MS);

      child.stdout.on('data', (chunk: Buffer) => {
        const text = chunk.toString('utf-8');
        stdoutBuf += text;
        send({ type: 'stdout', data: text });
      });
      child.stderr.on('data', (chunk: Buffer) => {
        const text = chunk.toString('utf-8');
        stderrBuf += text;
        send({ type: 'stderr', data: text });
      });

      child.on('error', err => {
        clearTimeout(timer);
        send({ type: 'error', message: err.message });
        controller.close();
      });

      child.on('close', (code, signal) => {
        clearTimeout(timer);
        send({
          type: 'exit',
          exitCode: code,
          signal,
          ok: !timedOut && code === 0,
          timedOut,
          result: parseResult(stdoutBuf),
          stderr: stderrBuf,
        });
        controller.close();
      });
    },
    cancel() {
      if (!child.killed) child.kill('SIGKILL');
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
    },
  });
};

export async function POST(request: Request) {
  let body: any;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const scriptPath = resolveScriptPath(body?.script);
  if (!scriptPath) {
    return NextResponse.json(
      { error: 'Invalid or unknown script. Must be a *.py file inside ui_scripts/.' },
      { status: 400 },
    );
  }

  const normalized = normalizeArgs(body?.args);
  if (!Array.isArray(normalized)) {
    return NextResponse.json({ error: normalized.error }, { status: 400 });
  }

  if (body?.stream === true) {
    return runStreaming(scriptPath, normalized);
  }

  const result = await runBuffered(scriptPath, normalized);
  const status = result.ok ? 200 : result.timedOut ? 504 : 500;
  return NextResponse.json(result, { status });
}
