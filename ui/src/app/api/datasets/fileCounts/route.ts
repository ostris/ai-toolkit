import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

const imageExtensions = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp']);
const audioExtensions = new Set(['.mp3', '.wav', '.flac', '.ogg']);

export async function POST(request: Request) {
  const datasetsRoot = await getDatasetsRoot();
  const body = await request.json();
  const datasetPath = body?.datasetPath;

  if (typeof datasetPath !== 'string' || datasetPath.trim() === '') {
    return NextResponse.json({ error: 'datasetPath is required' }, { status: 400 });
  }

  const normalizedRoot = path.resolve(datasetsRoot);
  const normalizedDatasetPath = path.resolve(datasetPath);
  const relativePath = path.relative(normalizedRoot, normalizedDatasetPath);
  const isOutsideRoot = relativePath.startsWith('..') || path.isAbsolute(relativePath);

  if (isOutsideRoot) {
    return NextResponse.json({ error: 'Dataset path must be inside the datasets folder' }, { status: 400 });
  }

  if (!fs.existsSync(normalizedDatasetPath)) {
    return NextResponse.json({ error: 'Dataset folder not found' }, { status: 404 });
  }

  try {
    const counts = countDatasetFiles(normalizedDatasetPath);
    return NextResponse.json(counts);
  } catch (error) {
    console.error('Error counting dataset files:', error);
    return NextResponse.json({ error: 'Failed to count dataset files' }, { status: 500 });
  }
}

function countDatasetFiles(dir: string) {
  let imageFileCount = 0;
  let audioFileCount = 0;
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    if (entry.name.startsWith('.')) {
      continue;
    }

    const entryPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      if (entry.name === '_controls') {
        continue;
      }

      const nestedCounts = countDatasetFiles(entryPath);
      imageFileCount += nestedCounts.imageFileCount;
      audioFileCount += nestedCounts.audioFileCount;
      continue;
    }

    if (!entry.isFile()) {
      continue;
    }

    const extension = path.extname(entry.name).toLowerCase();
    if (imageExtensions.has(extension)) {
      imageFileCount += 1;
    }
    if (audioExtensions.has(extension)) {
      audioFileCount += 1;
    }
  }

  return { imageFileCount, audioFileCount };
}