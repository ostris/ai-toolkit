import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const execFileAsync = promisify(execFile);

export async function POST(request: Request) {
  let concatListPath: string | null = null;
  try {
    const body = await request.json();
    const { videoPaths } = body;

    if (!Array.isArray(videoPaths) || videoPaths.length < 2) {
      return NextResponse.json({ error: 'At least 2 video paths are required' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();

    // Validate all paths
    for (const videoPath of videoPaths) {
      if (!videoPath.startsWith(datasetsPath) && !videoPath.startsWith(trainingPath)) {
        return NextResponse.json({ error: 'Invalid video path' }, { status: 400 });
      }

      // prevent path traversal
      if (videoPath.includes('..')) {
        return NextResponse.json({ error: 'Invalid video path' }, { status: 400 });
      }

      if (!/\.(mp4|avi|mov|mkv|wmv|m4v|flv)$/i.test(videoPath)) {
        return NextResponse.json({ error: 'Not a video file' }, { status: 400 });
      }

      if (!fs.existsSync(videoPath)) {
        return NextResponse.json({ error: `Video not found: ${videoPath}` }, { status: 404 });
      }
    }

    // Generate output path based on first video
    const firstVideo = videoPaths[0];
    const dir = path.dirname(firstVideo);
    const ext = path.extname(firstVideo);
    const base = path.basename(firstVideo, ext);

    let outputPath = path.join(dir, `${base}_merged${ext}`);
    let counter = 1;
    while (fs.existsSync(outputPath)) {
      outputPath = path.join(dir, `${base}_merged_${counter}${ext}`);
      counter++;
    }

    // Create a temporary concat list file
    concatListPath = path.join(os.tmpdir(), `concat_${Date.now()}.txt`);
    const concatContent = videoPaths.map(p => `file '${p.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`).join('\n');
    fs.writeFileSync(concatListPath, concatContent, 'utf8');

    // Run ffmpeg to concatenate
    await execFileAsync('ffmpeg', ['-f', 'concat', '-safe', '0', '-i', concatListPath, '-c', 'copy', outputPath]);

    // Delete source videos and their captions
    for (const videoPath of videoPaths) {
      fs.unlinkSync(videoPath);
      const captionPath = videoPath.replace(/\.[^/.]+$/, '') + '.txt';
      if (fs.existsSync(captionPath)) {
        fs.unlinkSync(captionPath);
      }
    }

    return NextResponse.json({ success: true, outputPath });
  } catch (error: any) {
    console.error('Error merging videos:', error);
    if (error.code === 'ENOENT') {
      return NextResponse.json({ error: 'ffmpeg is not installed or not found in PATH' }, { status: 500 });
    }
    const message = error.stderr ? `Failed to merge videos: ${error.stderr}` : 'Failed to merge videos';
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    if (concatListPath && fs.existsSync(concatListPath)) {
      try {
        fs.unlinkSync(concatListPath);
      } catch (_) {
        // ignore cleanup errors
      }
    }
  }
}
