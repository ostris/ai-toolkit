import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const execFileAsync = promisify(execFile);

export async function POST(request: Request) {
  let tempOutput: string | null = null;
  try {
    const body = await request.json();
    const { videoPath, startTime, endTime } = body;
    let datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();

    // make sure the path is in the datasets or training folder
    if (!videoPath.startsWith(datasetsPath) && !videoPath.startsWith(trainingPath)) {
      return NextResponse.json({ error: 'Invalid video path' }, { status: 400 });
    }

    // prevent path traversal
    if (videoPath.includes('..')) {
      return NextResponse.json({ error: 'Invalid video path' }, { status: 400 });
    }

    // make sure it is a video
    if (!/\.(mp4|avi|mov|mkv|wmv|m4v|flv)$/i.test(videoPath)) {
      return NextResponse.json({ error: 'Not a video file' }, { status: 400 });
    }

    // validate times
    const start = parseFloat(startTime);
    const end = parseFloat(endTime);
    if (isNaN(start) || isNaN(end) || start < 0 || end <= start) {
      return NextResponse.json({ error: 'Invalid start/end time' }, { status: 400 });
    }

    if (!fs.existsSync(videoPath)) {
      return NextResponse.json({ error: 'Video not found' }, { status: 404 });
    }

    const dir = path.dirname(videoPath);
    const ext = path.extname(videoPath);
    const base = path.basename(videoPath, ext);
    tempOutput = path.join(dir, `${base}_trimmed_temp${ext}`);

    await execFileAsync('ffmpeg', [
      '-i', videoPath,
      '-ss', `${start}`,
      '-to', `${end}`,
      '-c:v', 'libx264',
      '-c:a', 'aac',
      tempOutput,
    ]);

    // Replace original with trimmed
    fs.unlinkSync(videoPath);
    fs.renameSync(tempOutput, videoPath);
    tempOutput = null;

    return NextResponse.json({ success: true });
  } catch (error: any) {
    console.error('Error trimming video:', error);
    if (error.code === 'ENOENT') {
      return NextResponse.json({ error: 'ffmpeg is not installed or not found in PATH' }, { status: 500 });
    }
    const message = error.stderr ? `Failed to trim video: ${error.stderr}` : 'Failed to trim video';
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    if (tempOutput && fs.existsSync(tempOutput)) {
      try {
        fs.unlinkSync(tempOutput);
      } catch (_) {
        // ignore cleanup errors
      }
    }
  }
}
