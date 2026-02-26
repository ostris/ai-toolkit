import prisma from '../prisma';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getDatasetsRoot } from '../paths';

const MAX_CONCURRENT = 2;

// Statuses set by the user intentionally — don't overwrite on process close.
const STOPPED_STATUSES = new Set(['cancelled']);

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

const runDownload = (
  downloadId: string,
  url: string,
  outputDir: string,
  format: string,
  cookiesFile: string,
): void => {
  const ytDlpPath = getYtDlpPath();

  const args: string[] = [
    '--newline',
    '--merge-output-format', 'mp4',
    '-o', path.join(outputDir, '%(title)s.%(ext)s'),
  ];

  if (format && format.trim()) {
    args.push('-f', format.trim());
  }

  const trimmedCookies = cookiesFile?.trim();
  if (trimmedCookies && fs.existsSync(trimmedCookies)) {
    args.push('--cookies', trimmedCookies);
  }

  args.push(url);

  const proc = spawn(ytDlpPath, args);

  // Store PID so the API can terminate the process on cancel/pause
  if (proc.pid) {
    prisma.videoDownload
      .update({ where: { id: downloadId }, data: { pid: proc.pid } })
      .catch(err => console.error(`[downloads] Failed to store PID for ${downloadId}:`, err));
  }

  let filename = '';
  let stdoutBuffer = '';
  let stderrBuffer = '';
  let lastError = '';

  // yt-dlp progress line: [download]   1.2% of   150.23MiB at   2.34MiB/s ETA 01:03:45
  const progressRe = /\[download\]\s+(\d+\.?\d*)%\s+of\s+([\d.]+\s*\S+)\s+at\s+([\d.]+\s*\S+\/s)/;

  const processLine = (line: string) => {
    const progressMatch = line.match(progressRe);
    if (progressMatch) {
      const progress = parseFloat(progressMatch[1]);
      const filesize = progressMatch[2].trim();
      const speed = progressMatch[3].trim();
      prisma.videoDownload
        .update({ where: { id: downloadId }, data: { progress, filesize, speed } })
        .catch(err => console.error(`[downloads] Failed to update progress for ${downloadId}:`, err));
      return;
    }

    // Simpler fallback for lines that only have percentage (no size/speed yet)
    const simpleProgress = line.match(/\[download\]\s+(\d+\.?\d*)%/);
    if (simpleProgress) {
      const progress = parseFloat(simpleProgress[1]);
      prisma.videoDownload
        .update({ where: { id: downloadId }, data: { progress } })
        .catch(err => console.error(`[downloads] Failed to update progress for ${downloadId}:`, err));
    }

    const filenameMatch = line.match(/\[download\] Destination: (.+)/);
    if (filenameMatch) {
      filename = path.basename(filenameMatch[1].trim());
    }
  };

  proc.stdout.on('data', (data: Buffer) => {
    stdoutBuffer += data.toString();
    const lines = stdoutBuffer.split('\n');
    stdoutBuffer = lines.pop() ?? '';
    for (const line of lines) {
      processLine(line);
    }
  });

  proc.stderr.on('data', (data: Buffer) => {
    stderrBuffer += data.toString();
    const lines = stderrBuffer.split('\n');
    stderrBuffer = lines.pop() ?? '';
    for (const line of lines) {
      processLine(line);
      if (line.trim()) {
        lastError = line.trim();
      }
    }
  });

  proc.on('close', async (code: number | null) => {
    // Always clear the PID and speed — process is no longer running
    await prisma.videoDownload
      .update({ where: { id: downloadId }, data: { pid: null, speed: '' } })
      .catch(err => console.error(`[downloads] Failed to clear PID for ${downloadId}:`, err));

    // Don't overwrite a status the user set intentionally (cancelled/paused)
    const current = await prisma.videoDownload
      .findUnique({ where: { id: downloadId }, select: { status: true } })
      .catch(() => null);

    if (current && STOPPED_STATUSES.has(current.status)) {
      return;
    }

    if (code === 0) {
      prisma.videoDownload
        .update({ where: { id: downloadId }, data: { status: 'completed', progress: 100, filename } })
        .catch(err => console.error(`[downloads] Failed to mark completed for ${downloadId}:`, err));
    } else {
      const errorMsg = lastError || `yt-dlp exited with code ${code}`;
      prisma.videoDownload
        .update({ where: { id: downloadId }, data: { status: 'failed', error: errorMsg } })
        .catch(err => console.error(`[downloads] Failed to mark failed for ${downloadId}:`, err));
    }
  });

  proc.on('error', (err: Error) => {
    prisma.videoDownload
      .update({ where: { id: downloadId }, data: { status: 'failed', speed: '', pid: null, error: err.message } })
      .catch(dbErr => console.error(`[downloads] Failed to mark error for ${downloadId}:`, dbErr));
  });
};

export default async function processDownloads() {
  const downloadingCount = await prisma.videoDownload.count({
    where: { status: 'downloading' },
  });

  if (downloadingCount >= MAX_CONCURRENT) return;

  const toStart = MAX_CONCURRENT - downloadingCount;

  const pending = await prisma.videoDownload.findMany({
    where: { status: 'pending' },
    orderBy: { created_at: 'asc' },
    take: toStart,
  });

  if (pending.length === 0) return;

  const datasetsRoot = await getDatasetsRoot();

  for (const download of pending) {
    const outputDir = path.join(datasetsRoot, download.dataset);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Mark as downloading first to prevent re-pickup on next iteration
    await prisma.videoDownload.update({
      where: { id: download.id },
      data: { status: 'downloading', progress: 0 },
    });

    // Start download asynchronously
    runDownload(download.id, download.url, outputDir, download.format, download.cookies_file);
  }
}
