import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

const isWindows = process.platform === 'win32';

function getToolkitRoot(): string {
  return path.resolve(process.cwd(), '..');
}

function findPython(toolkitRoot: string): string {
  const venvDirs = ['.venv', 'venv', 'app/env'];
  for (const vdir of venvDirs) {
    const full = path.join(toolkitRoot, vdir);
    if (fs.existsSync(full)) {
      const exe = isWindows
        ? path.join(full, 'Scripts', 'python.exe')
        : path.join(full, 'bin', 'python');
      if (fs.existsSync(exe)) return exe;
    }
  }
  return 'python';
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { lora_1_path, lora_2_path, output_name, dare_drop_rate } = body;

    if (!lora_1_path || !lora_2_path || !output_name) {
      return NextResponse.json({ error: 'lora_1_path, lora_2_path, and output_name are required' }, { status: 400 });
    }

    const toolkitRoot = getToolkitRoot();
    const runFile = path.join(toolkitRoot, 'run.py');
    if (!fs.existsSync(runFile)) {
      return NextResponse.json({ error: `run.py not found at ${runFile}` }, { status: 500 });
    }

    const pythonPath = findPython(toolkitRoot);
    const outputDir = path.join(toolkitRoot, 'output');
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    const outputPath = `output/${output_name}.safetensors`;
    const mergeDir = path.join(outputDir, '_merge_jobs');
    if (!fs.existsSync(mergeDir)) fs.mkdirSync(mergeDir, { recursive: true });

    const jobId = `merge_${Date.now()}`;
    const configPath = path.join(mergeDir, `${jobId}.json`);
    const logPath = path.join(mergeDir, `${jobId}.log`);

    const jobConfig = {
      job: 'MergeJob',
      config: {
        name: output_name,
        process: [
          {
            type: 'merge_orthogonal',
            lora_1_path,
            lora_2_path,
            output_path: outputPath,
            dare_drop_rate: dare_drop_rate ?? 0.5,
            device: 'cuda',
          },
        ],
      },
      meta: { name: output_name, version: '1.0' },
    };

    fs.writeFileSync(configPath, JSON.stringify(jobConfig, null, 2));

    const args = [runFile, configPath, '--log', logPath];

    const subprocess = spawn(pythonPath, args, {
      env: {
        ...process.env,
        CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES: '0',
      },
      cwd: toolkitRoot,
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      ...(isWindows ? { windowsHide: true } : {}),
    });

    let lastOutput = '';
    subprocess.stdout?.on('data', (data: Buffer) => {
      lastOutput = data.toString().trim();
    });
    subprocess.stderr?.on('data', (data: Buffer) => {
      lastOutput = data.toString().trim();
    });

    const statusFile = path.join(mergeDir, `${jobId}.status`);
    fs.writeFileSync(statusFile, JSON.stringify({ status: 'running', pid: subprocess.pid, started: Date.now() }));

    subprocess.on('close', (code: number | null) => {
      const finalStatus = code === 0 ? 'completed' : 'error';
      try {
        fs.writeFileSync(
          statusFile,
          JSON.stringify({ status: finalStatus, exitCode: code, output: lastOutput, finished: Date.now() }),
        );
      } catch (_) {}
    });

    subprocess.on('error', (err: Error) => {
      try {
        fs.writeFileSync(
          statusFile,
          JSON.stringify({ status: 'error', error: err.message, finished: Date.now() }),
        );
      } catch (_) {}
    });

    if (subprocess.unref) subprocess.unref();

    return NextResponse.json({
      status: 'started',
      jobId,
      outputPath: path.join(toolkitRoot, outputPath),
      statusFile,
      logPath,
      pid: subprocess.pid,
    });
  } catch (error: any) {
    console.error('Merge run error:', error);
    return NextResponse.json({ error: error?.message || 'Failed to start merge' }, { status: 500 });
  }
}
