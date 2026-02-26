import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getDatasetsRoot } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';

const execFileAsync = promisify(execFile);

const SCRIPT_PATH = path.join(TOOLKIT_ROOT, 'scripts', 'caption_image.py');
const PYTHON_EXECUTABLE = process.env.PYTHON_EXECUTABLE || 'python3';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, triggerWord, systemPrompt } = body;

    if (!imgPath || typeof imgPath !== 'string') {
      return NextResponse.json({ error: 'imgPath is required' }, { status: 400 });
    }

    // Security: prevent path traversal
    if (imgPath.includes('..')) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();

    // Security: ensure path is within allowed directory
    if (!imgPath.startsWith(datasetsPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ error: 'File does not exist' }, { status: 404 });
    }

    if (!fs.existsSync(SCRIPT_PATH)) {
      return NextResponse.json({ error: 'Captioning script not found' }, { status: 500 });
    }

    const args = [
      SCRIPT_PATH,
      '--img_path', imgPath,
      '--trigger_word', (triggerWord || '').toString(),
      '--system_prompt', (systemPrompt || '').toString(),
    ];

    const { stdout } = await execFileAsync(PYTHON_EXECUTABLE, args, {
      timeout: 300000, // 5 minutes
    });

    let result: { caption?: string; error?: string };
    try {
      result = JSON.parse(stdout.trim());
    } catch {
      return NextResponse.json({ error: 'Failed to parse captioning output' }, { status: 500 });
    }

    if (result.error) {
      return NextResponse.json({ error: result.error }, { status: 500 });
    }

    return NextResponse.json({ caption: result.caption || '' });
  } catch (error: any) {
    console.error('Error running AI captioning:', error);
    const message = error?.stderr ? `Captioning failed: ${error.stderr}` : 'Captioning failed';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
