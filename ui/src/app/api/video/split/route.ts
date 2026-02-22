import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const execFileAsync = promisify(execFile);

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { videoPath, secondsPerSegment } = body;
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

    // validate seconds per segment
    const seconds = parseInt(secondsPerSegment, 10);
    if (isNaN(seconds) || seconds < 1) {
      return NextResponse.json({ error: 'Invalid seconds per segment' }, { status: 400 });
    }

    if (!fs.existsSync(videoPath)) {
      return NextResponse.json({ error: 'Video not found' }, { status: 404 });
    }

    const dir = path.dirname(videoPath);
    const ext = path.extname(videoPath);
    const base = path.basename(videoPath, ext);
    const outputPattern = path.join(dir, `${base}_%03d${ext}`);

    await execFileAsync('ffmpeg', [
      '-i', videoPath,
      '-c', 'copy',
      '-map', '0',
      '-segment_time', `${seconds}`,
      '-f', 'segment',
      '-reset_timestamps', '1',
      outputPattern,
    ]);

    // delete the original video
    fs.unlinkSync(videoPath);

    // delete associated caption file if it exists
    const captionPath = videoPath.replace(/\.[^/.]+$/, '') + '.txt';
    if (fs.existsSync(captionPath)) {
      fs.unlinkSync(captionPath);
    }

    return NextResponse.json({ success: true });
  } catch (error: any) {
    console.error('Error splitting video:', error);
    if (error.code === 'ENOENT') {
      return NextResponse.json({ error: 'ffmpeg is not installed or not found in PATH' }, { status: 500 });
    }
    const message = error.stderr ? `Failed to split video: ${error.stderr}` : 'Failed to split video';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
