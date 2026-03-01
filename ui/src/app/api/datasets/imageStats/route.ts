import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getDatasetsRoot } from '@/server/settings';
import { videoExtensions } from '@/utils/basic';

const execFileAsync = promisify(execFile);

interface ImageStats {
  totalCount: number;
  imageCount: number;
  videoCount: number;
  totalVideoDuration: number;
  resolutionBreakdown: { [resolution: string]: number };
}

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
  const datasetsPath = await getDatasetsRoot();
  const { searchParams } = new URL(request.url);
  const datasetName = searchParams.get('datasetName');

  // Validate datasetName
  if (!datasetName || typeof datasetName !== 'string' || datasetName.trim() === '') {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  // Prevent path traversal attacks
  if (datasetName.includes('..') || datasetName.includes('/') || datasetName.includes('\\')) {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  const datasetFolder = path.join(datasetsPath, datasetName);

  // Verify the resolved path is within datasetsPath
  if (!datasetFolder.startsWith(datasetsPath)) {
    return NextResponse.json({ error: 'Invalid dataset path' }, { status: 400 });
  }

  // Check if folder exists
  if (!fs.existsSync(datasetFolder)) {
    return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
  }

  // Initialize stats with defaults
  let totalCount = 0;
  let imageCount = 0;
  let videoCount = 0;
  let totalVideoDuration = 0;
  const resolutionBreakdown: { [resolution: string]: number } = {};
  let hasError = false;

  try {
    // Find all images recursively
    const imageFiles = findImagesRecursively(datasetFolder);
    totalCount = imageFiles.length;

    // Separate video files from image files in a single pass
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

    // Get video durations concurrently
    const CONCURRENCY_LIMIT = 10;
    for (let i = 0; i < videoFiles.length; i += CONCURRENCY_LIMIT) {
      const batch = videoFiles.slice(i, i + CONCURRENCY_LIMIT);
      const durations = await Promise.all(batch.map(vp => getVideoDuration(vp)));
      totalVideoDuration += durations.reduce((sum, d) => sum + d, 0);
    }

    // Get resolution for each image with concurrent processing
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
          } catch (error) {
            console.error(`Error reading image metadata for ${imgPath}:`, error);
            // If we can't read the image, count it as unknown
            const unknownKey = 'unknown resolution';
            resolutionBreakdown[unknownKey] = (resolutionBreakdown[unknownKey] || 0) + 1;
          }
        })
      );
    }
  } catch (error) {
    console.error('Error calculating image stats:', error);
    hasError = true;
  }

  // Always return stats with what we have, even if there were errors
  const stats: ImageStats = {
    totalCount,
    imageCount,
    videoCount,
    totalVideoDuration,
    resolutionBreakdown,
  };

  return NextResponse.json(stats);
}

/**
 * Recursively finds all image files in a directory and its subdirectories
 * @param dir Directory to search
 * @returns Array of absolute paths to image files
 */
function findImagesRecursively(dir: string): string[] {
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'];
  let results: string[] = [];

  const items = fs.readdirSync(dir);

  for (const item of items) {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);

    if (stat.isDirectory() && item !== '_controls' && !item.startsWith('.')) {
      // If it's a directory, recursively search it
      results = results.concat(findImagesRecursively(itemPath));
    } else {
      // If it's a file, check if it's an image and not trashed
      const ext = path.extname(itemPath).toLowerCase();
      if (imageExtensions.includes(ext) && !item.startsWith('trash_')) {
        results.push(itemPath);
      }
    }
  }

  return results;
}
