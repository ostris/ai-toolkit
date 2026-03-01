import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getDatasetsRoot } from '@/server/settings';
import { videoExtensions, imgExtensions, audioExtensions } from '@/utils/basic';

const execFileAsync = promisify(execFile);

export interface ImageMetadataEntry {
  img_path: string;
  duration?: number;
  width?: number;
  height?: number;
  scores?: Record<string, number>;
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

function readScores(imgPath: string): Record<string, number> {
  const parsed = path.parse(imgPath);
  const csvPath = path.format({ dir: parsed.dir, name: parsed.name, ext: '.csv' });
  if (!fs.existsSync(csvPath)) return {};
  try {
    const content = fs.readFileSync(csvPath, 'utf-8');
    const scores: Record<string, number> = {};
    const lines = content.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const commaIndex = trimmed.indexOf(',');
      if (commaIndex === -1) continue;
      const metric = trimmed.slice(0, commaIndex).trim();
      const valueStr = trimmed.slice(commaIndex + 1).trim();
      const value = parseFloat(valueStr);
      if (metric && !isNaN(value)) {
        scores[metric] = value;
      }
    }
    return scores;
  } catch {
    return {};
  }
}

function findImagesRecursively(dir: string): string[] {
  const allExtensions = [...imgExtensions, ...videoExtensions, ...audioExtensions];
  let results: string[] = [];
  const items = fs.readdirSync(dir, { withFileTypes: true });
  for (const item of items) {
    const itemPath = path.join(dir, item.name);
    if (item.isDirectory() && item.name !== '_controls' && !item.name.startsWith('.')) {
      results = results.concat(findImagesRecursively(itemPath));
    } else if (item.isFile()) {
      const ext = path.extname(itemPath).toLowerCase();
      if (allExtensions.includes(ext) && !item.name.startsWith('trash_')) {
        results.push(itemPath);
      }
    }
  }
  return results;
}

export async function GET(request: Request) {
  const datasetsPath = await getDatasetsRoot();
  const { searchParams } = new URL(request.url);
  const datasetName = searchParams.get('datasetName');

  if (!datasetName || typeof datasetName !== 'string' || datasetName.trim() === '') {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  if (datasetName.includes('..') || datasetName.includes('/') || datasetName.includes('\\')) {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  const datasetFolder = path.join(datasetsPath, datasetName);

  if (!datasetFolder.startsWith(datasetsPath)) {
    return NextResponse.json({ error: 'Invalid dataset path' }, { status: 400 });
  }

  if (!fs.existsSync(datasetFolder)) {
    return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
  }

  const imageFiles = findImagesRecursively(datasetFolder);
  const CONCURRENCY_LIMIT = 10;
  const images: ImageMetadataEntry[] = new Array(imageFiles.length);

  for (let i = 0; i < imageFiles.length; i += CONCURRENCY_LIMIT) {
    const batch = imageFiles.slice(i, i + CONCURRENCY_LIMIT);
    await Promise.all(
      batch.map(async (imgPath, batchIndex) => {
        const index = i + batchIndex;
        const ext = path.extname(imgPath).toLowerCase();
        if (videoExtensions.includes(ext)) {
          const duration = await getVideoDuration(imgPath);
          images[index] = { img_path: imgPath, duration };
        } else {
          const meta: ImageMetadataEntry = { img_path: imgPath };
          try {
            const sharpMeta = await sharp(imgPath).metadata();
            meta.width = sharpMeta.width;
            meta.height = sharpMeta.height;
          } catch {
            // ignore metadata read errors
          }
          const scores = readScores(imgPath);
          if (Object.keys(scores).length > 0) {
            meta.scores = scores;
          }
          images[index] = meta;
        }
      }),
    );
  }

  return NextResponse.json({ images });
}
