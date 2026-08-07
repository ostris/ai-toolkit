/**
 * POST /api/datasets/remote/[source]/import
 *
 * Generic import endpoint for any registered data-source plugin.
 * Asks the Python plugin for its job config via the `job-config` CLI command,
 * then spawns run.py and streams progress back as Server-Sent Events.
 *
 * Expected body:
 *   source_type   string  — group id from browse() (e.g. "character", "person")
 *   source_id     string  — item id from browse()
 *   trigger_word  string  — written into caption .txt files
 *   dataset_name  string  — output folder name under datasets/
 *   overwrite     bool    — re-download even if already cached
 *   ...           any source-specific fields declared by get_import_fields()
 *
 * SSE event types:
 *   total     { count: number }
 *   progress  { done: number, total: number }
 *   complete  { downloaded: number }
 *   error     { message: string }
 */
import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import * as readline from 'readline';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';

const isWindows = process.platform === 'win32';

function getPythonPath(): string {
  for (const dir of ['.venv', 'venv']) {
    const p = path.join(TOOLKIT_ROOT, dir, isWindows ? 'Scripts/python.exe' : 'bin/python');
    if (existsSync(p)) return p;
  }
  return 'python';
}

/** Ask the Python plugin for its process config by writing params to stdin. */
function getJobConfig(
  source: string,
  params: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      getPythonPath(),
      ['-m', 'toolkit.dataset_sources.cli', 'job-config', source],
      { cwd: TOOLKIT_ROOT, stdio: ['pipe', 'pipe', 'pipe'] },
    );

    let out = '';
    let err = '';
    child.stdout.on('data', (d: Buffer) => (out += d.toString()));
    child.stderr.on('data', (d: Buffer) => (err += d.toString()));

    child.stdin.write(JSON.stringify(params));
    child.stdin.end();

    child.on('close', code => {
      try {
        const data = JSON.parse(out);
        if (data.error || code !== 0) {
          reject(new Error(data.error || err.trim() || 'job-config failed'));
        } else {
          resolve(data);
        }
      } catch {
        reject(new Error(`Invalid job-config response: ${out}`));
      }
    });
  });
}

export async function POST(
  request: NextRequest,
  { params }: { params: { source: string } },
) {
  const { source } = params;
  const body = await request.json() as Record<string, unknown>;
  const { source_type, source_id } = body;

  if (!source_type || source_id == null) {
    return NextResponse.json({ error: 'source_type and source_id are required' }, { status: 400 });
  }

  let processConfig: Record<string, unknown>;
  try {
    processConfig = await getJobConfig(source, body);
  } catch (e: unknown) {
    return NextResponse.json({ error: String(e) }, { status: 400 });
  }

  const jobConfig = {
    job: 'extension',
    config: {
      name: `${source}_import_${source_type}_${source_id}`,
      process: [processConfig],
    },
  };

  const tmpDir = path.join(TOOLKIT_ROOT, 'output', `.${source}_tmp`);
  mkdirSync(tmpDir, { recursive: true });
  const configPath = path.join(
    tmpDir,
    `fetch_${source_type}_${source_id}_${Date.now()}.json`,
  );
  writeFileSync(configPath, JSON.stringify(jobConfig, null, 2));

  const pythonPath = getPythonPath();
  const runFilePath = path.join(TOOLKIT_ROOT, 'run.py');

  if (!existsSync(runFilePath)) {
    return NextResponse.json({ error: 'run.py not found' }, { status: 500 });
  }

  const encoder = new TextEncoder();
  const sseEvent = (type: string, data: object) =>
    encoder.encode(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);

  const stream = new ReadableStream({
    start(controller) {
      const child = spawn(pythonPath, [runFilePath, configPath], {
        stdio: ['ignore', 'pipe', 'pipe'],
        cwd: TOOLKIT_ROOT,
      });

      let currentTotal = 0;
      const errorLines: string[] = [];

      const processLine = (line: string, isStderr: boolean) => {
        if (isStderr) errorLines.push(line);
        const progressMatch = line.match(/^PROGRESS:(\d+)\/(\d+)$/);
        if (progressMatch) {
          currentTotal = parseInt(progressMatch[2]);
          controller.enqueue(
            sseEvent('progress', {
              done: parseInt(progressMatch[1]),
              total: currentTotal,
            }),
          );
          return;
        }
        const totalMatch = line.match(/— (\d+) (?:picture|image)\(s\) (?:found|downloaded)/);
        if (totalMatch) {
          currentTotal = parseInt(totalMatch[1]);
          controller.enqueue(sseEvent('total', { count: currentTotal }));
          return;
        }
        const doneMatch = line.match(/Done — (\d+) downloaded/);
        if (doneMatch) {
          controller.enqueue(sseEvent('complete', { downloaded: parseInt(doneMatch[1]) }));
        }
      };

      readline.createInterface({ input: child.stdout! }).on('line', line => processLine(line, false));
      readline.createInterface({ input: child.stderr! }).on('line', line => processLine(line, true));

      child.on('close', code => {
        if (code !== 0) {
          const errMsg = errorLines.slice(-10).join('\n').trim() || `Process exited with code ${code}`;
          controller.enqueue(sseEvent('error', { message: errMsg }));
        }
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}
