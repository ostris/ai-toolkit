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

  if (!fs.existsSync(normalizedPath) || !fs.statSync(normalizedPath).isDirectory()) {
    return NextResponse.json({ error: 'Folder not found' }, { status: 404 });
  }

  let totalCount = 0;
  let imageCount = 0;
  let videoCount = 0;
  let totalVideoDuration = 0;
  const resolutionBreakdown: { [resolution: string]: number } = {};

  try {
    const imageFiles = findImagesInFolder(normalizedPath);
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

function findImagesInFolder(dir: string): string[] {
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.mp3', '.wav'];
  let results: string[] = [];

  const items = fs.readdirSync(dir, { withFileTypes: true });
  for (const item of items) {
    if (item.isDirectory() && !item.name.startsWith('.')) {
      const subPath = path.join(dir, item.name);
      results = results.concat(findImagesInFolder(subPath));
    } else if (item.isFile()) {
      const itemPath = path.join(dir, item.name);
      const ext = path.extname(item.name).toLowerCase();
      if (imageExtensions.includes(ext) && !item.name.startsWith('trash_')) {
        results.push(itemPath);
      }
    }
  }

  return results;
}
