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
    const { videoPath } = body;
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

    if (!fs.existsSync(videoPath)) {
      return NextResponse.json({ error: 'Video not found' }, { status: 404 });
    }

    const dir = path.dirname(videoPath);
    const base = path.basename(videoPath, path.extname(videoPath));
    const audioPath = path.join(dir, `${base}_audio.mp3`);

    await execFileAsync('ffmpeg', [
      '-y',
      '-i', videoPath,
      '-vn',
      '-acodec', 'libmp3lame',
      '-q:a', '2',
      audioPath,
    ]);

    return NextResponse.json({ success: true, audioPath });
  } catch (error: any) {
    console.error('Error extracting audio:', error);
    if (error.code === 'ENOENT') {
      return NextResponse.json({ error: 'ffmpeg is not installed or not found in PATH' }, { status: 500 });
    }
    const message = error.stderr ? `Failed to extract audio: ${error.stderr}` : 'Failed to extract audio';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
