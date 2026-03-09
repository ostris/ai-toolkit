import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { videoExtensions } from '@/utils/basic';

const execFileAsync = promisify(execFile);

async function getVideoDuration(videoPath: string): Promise<number> {
  try {
    const { stdout } = await execFileAsync('ffprobe', [
      '-v', 'error',
      '-show_entries', 'format=duration',
      '-of', 'default=noprint_wrappers=1:nokey=1',
      videoPath,
    ]);
    const duration = parseFloat(stdout.trim());
    return isNaN(duration) ? 0 : duration;
  } catch {
    return 0;
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const folderPath = searchParams.get('folderPath');

  if (!folderPath || typeof folderPath !== 'string') {
    return NextResponse.json({ error: 'folderPath is required' }, { status: 400 });
  }

  // Prevent path traversal
  const normalizedPath = path.normalize(folderPath);
  if (normalizedPath.includes('..')) {
    return NextResponse.json({ error: 'Invalid folder path' }, { status: 400 });
  }

  try {
    const stat = await fs.promises.stat(normalizedPath);
    if (!stat.isDirectory()) {
      return NextResponse.json({ error: 'Folder not found' }, { status: 404 });
    }
  } catch {
    return NextResponse.json({ error: 'Folder not found' }, { status: 404 });
  }

  let totalCount = 0;
  let imageCount = 0;
  let videoCount = 0;
  let totalVideoDuration = 0;
  const resolutionBreakdown: { [resolution: string]: number } = {};

  try {
    const imageFiles = await findImagesInFolder(normalizedPath);
    totalCount = imageFiles.length;

    const videoFiles: string[] = [];
    const nonVideoFiles: string[] = [];
    for (const f of imageFiles) {
      if (videoExtensions.includes(path.extname(f).toLowerCase())) {
        videoFiles.push(f);
      } else {
        nonVideoFiles.push(f);
      }
    }
    imageCount = nonVideoFiles.length;
    videoCount = videoFiles.length;

    const CONCURRENCY_LIMIT = 10;
    for (let i = 0; i < videoFiles.length; i += CONCURRENCY_LIMIT) {
      const batch = videoFiles.slice(i, i + CONCURRENCY_LIMIT);
      const durations = await Promise.all(batch.map(vp => getVideoDuration(vp)));
      totalVideoDuration += durations.reduce((sum, d) => sum + d, 0);
    }

    for (let i = 0; i < nonVideoFiles.length; i += CONCURRENCY_LIMIT) {
      const batch = nonVideoFiles.slice(i, i + CONCURRENCY_LIMIT);
      await Promise.allSettled(
        batch.map(async imgPath => {
          try {
            const metadata = await sharp(imgPath).metadata();
            const width = metadata.width || 0;
            const height = metadata.height || 0;
            const resolution = `${width}x${height}`;
            resolutionBreakdown[resolution] = (resolutionBreakdown[resolution] || 0) + 1;
          } catch {
            const unknownKey = 'unknown resolution';
            resolutionBreakdown[unknownKey] = (resolutionBreakdown[unknownKey] || 0) + 1;
          }
        })
      );
    }
  } catch (error) {
    console.error('Error calculating gallery image stats:', error);
  }

  return NextResponse.json({ totalCount, imageCount, videoCount, totalVideoDuration, resolutionBreakdown });
}

async function findImagesInFolder(dir: string): Promise<string[]> {
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.mp3', '.wav'];

  const items = await fs.promises.readdir(dir, { withFileTypes: true });
  const nestedResults = await Promise.all(
    items.map(async item => {
      if (item.isDirectory() && !item.name.startsWith('.')) {
        return findImagesInFolder(path.join(dir, item.name));
      } else if (item.isFile()) {
        const ext = path.extname(item.name).toLowerCase();
        if (imageExtensions.includes(ext) && !item.name.startsWith('trash_')) {
          return [path.join(dir, item.name)];
        }
      }
      return [];
    })
  );

  return nestedResults.flat();
}
