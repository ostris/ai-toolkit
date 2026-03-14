import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

function findCaptionFiles(dir: string): string[] {
  let results: string[] = [];
  const items = fs.readdirSync(dir, { withFileTypes: true });
  for (const item of items) {
    const itemPath = path.join(dir, item.name);
    if (item.isDirectory() && item.name !== '_controls' && !item.name.startsWith('.')) {
      results = results.concat(findCaptionFiles(itemPath));
    } else if (item.isFile() && item.name.endsWith('.txt') && !item.name.startsWith('trash_')) {
      results.push(itemPath);
    }
  }
  return results;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { datasetName } = body;

    if (!datasetName || typeof datasetName !== 'string') {
      return NextResponse.json({ error: 'datasetName is required' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    const datasetFolder = path.join(datasetsPath, datasetName);

    if (!datasetFolder.startsWith(datasetsPath) || datasetName.includes('..')) {
      return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
    }

    if (!fs.existsSync(datasetFolder) || !fs.statSync(datasetFolder).isDirectory()) {
      return NextResponse.json({ error: 'Dataset not found' }, { status: 404 });
    }

    const captionFiles = findCaptionFiles(datasetFolder);
    let deleted = 0;
    for (const file of captionFiles) {
      try {
        fs.unlinkSync(file);
        deleted++;
      } catch {
        // skip files that can't be deleted
      }
    }

    return NextResponse.json({ success: true, deleted });
  } catch (error) {
    console.error('Error clearing captions:', error);
    return NextResponse.json({ error: 'Failed to clear captions' }, { status: 500 });
  }
}
