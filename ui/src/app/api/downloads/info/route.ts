import { NextResponse } from 'next/server';
import { execFile } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs';

const execFileAsync = promisify(execFile);

const TOOLKIT_ROOT = path.resolve(process.cwd(), '..', '..');

const getYtDlpPath = (): string => {
  const isWindows = process.platform === 'win32';
  const venvDirs = ['.venv', 'venv'];
  for (const venvDir of venvDirs) {
    const candidate = isWindows
      ? path.join(TOOLKIT_ROOT, venvDir, 'Scripts', 'yt-dlp.exe')
      : path.join(TOOLKIT_ROOT, venvDir, 'bin', 'yt-dlp');
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return 'yt-dlp';
};

interface YtDlpFormat {
  format_id: string;
  ext: string;
  height?: number;
  width?: number;
  tbr?: number;
  filesize?: number;
  filesize_approx?: number;
  vcodec?: string;
  acodec?: string;
  format_note?: string;
}

export interface VideoInfo {
  title: string;
  duration: number | null;
  thumbnail: string | null;
  resolutions: { label: string; format: string }[];
}

function buildResolutions(formats: YtDlpFormat[]): { label: string; format: string }[] {
  // Collect unique heights from video-containing formats
  const heights = new Set<number>();
  for (const f of formats) {
    if (f.height && f.vcodec && f.vcodec !== 'none') {
      heights.add(f.height);
    }
  }

  const sorted = Array.from(heights).sort((a, b) => b - a);

  const options: { label: string; format: string }[] = [
    { label: 'Best quality', format: 'bestvideo+bestaudio/best' },
  ];

  for (const h of sorted) {
    options.push({
      label: `${h}p`,
      format: `bestvideo[height<=${h}]+bestaudio/best[height<=${h}]`,
    });
  }

  options.push({ label: 'Audio only', format: 'bestaudio/best' });

  return options;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const url = searchParams.get('url');

  if (!url || !url.trim()) {
    return NextResponse.json({ error: 'URL is required' }, { status: 400 });
  }

  const ytDlpPath = getYtDlpPath();

  try {
    const { stdout } = await execFileAsync(ytDlpPath, ['-j', '--no-playlist', url.trim()], {
      timeout: 30000,
    });

    const info = JSON.parse(stdout);
    const formats: YtDlpFormat[] = info.formats ?? [];

    const result: VideoInfo = {
      title: info.title ?? '',
      duration: info.duration ?? null,
      thumbnail: info.thumbnail ?? null,
      resolutions: buildResolutions(formats),
    };

    return NextResponse.json(result);
  } catch (error: any) {
    const stderr = error.stderr ? String(error.stderr).split('\n').filter(Boolean).pop() : undefined;
    const message = stderr || error.message || 'Failed to fetch video info';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
