import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import type { FileHandle } from 'fs/promises';
import { getTrainingFolder } from '@/server/settings';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'log.txt');

  try {
    await fs.promises.access(logPath);
  } catch {
    return NextResponse.json({ log: '', offset: 0, reset: true });
  }

  // Cap on the initial payload. The client renders the log through a terminal
  // emulator that collapses \r/cursor-movement rewrites (progress bars), so a
  // small newline-counted tail would be mostly bar churn that collapses to a
  // few rendered lines — send a generous byte tail instead and let the
  // emulator (which caps its own scrollback) do the trimming.
  const MAX_TAIL_BYTES = 5 * 1024 * 1024;
  // Client sends the byte offset it has already consumed so we only return new
  // content. `offset` omitted (or NaN) => initial load / full tail.
  const offsetParam = request.nextUrl.searchParams.get('offset');
  const offset = offsetParam === null ? NaN : parseInt(offsetParam, 10);

  const readRange = async (fh: FileHandle, start: number, end: number): Promise<string> => {
    const length = end - start;
    if (length <= 0) return '';
    const buffer = Buffer.alloc(length);
    await fh.read(buffer, 0, length, start);
    return buffer.toString('utf-8');
  };

  try {
    const stats = await fs.promises.stat(logPath);
    const size = stats.size;
    // If the client's offset is past the current end, the log was reset/truncated
    // (e.g. a fresh run overwrote it) — fall back to a fresh tail load.
    const isReset = Number.isNaN(offset) || offset > size;

    const fh = await fs.promises.open(logPath, 'r');
    try {
      if (isReset) {
        // Read only the tail of very large files to bound memory/payload.
        const start = Math.max(0, size - MAX_TAIL_BYTES);
        let log = await readRange(fh, start, size);
        // Drop a partial first line if we started mid-file.
        if (start > 0) {
          const newlineIdx = log.indexOf('\n');
          if (newlineIdx !== -1) {
            log = log.slice(newlineIdx + 1);
          }
        }
        return NextResponse.json({ log, offset: size, reset: true });
      }
      // Incremental: return only the bytes appended since the last offset.
      const log = await readRange(fh, offset, size);
      return NextResponse.json({ log, offset: size, reset: false });
    } finally {
      await fh.close();
    }
  } catch (error) {
    console.error('Error reading log file:', error);
    return NextResponse.json({ log: 'Error reading log file', offset: 0, reset: true });
  }
}
