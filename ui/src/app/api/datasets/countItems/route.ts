import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { countDatasetItemsRecursively, isPathInRoot } from '@/server/datasets';
import { getDatasetsRoot } from '@/server/settings';

type DatasetItemCounts = Record<string, number | null>;

export async function POST(request: Request) {
  try {
    const { datasetPaths } = await request.json();

    if (!Array.isArray(datasetPaths)) {
      return NextResponse.json({ error: 'datasetPaths must be an array' }, { status: 400 });
    }

    const datasetsRoot = await getDatasetsRoot();
    const counts: DatasetItemCounts = {};

    for (const rawDatasetPath of datasetPaths) {
      if (typeof rawDatasetPath !== 'string' || rawDatasetPath.trim() === '') continue;

      const datasetPath = path.resolve(rawDatasetPath);

      if (!isPathInRoot(datasetPath, datasetsRoot)) {
        counts[rawDatasetPath] = null;
        continue;
      }

      if (!fs.existsSync(datasetPath) || !fs.statSync(datasetPath).isDirectory()) {
        counts[rawDatasetPath] = null;
        continue;
      }

      counts[rawDatasetPath] = countDatasetItemsRecursively(datasetPath);
    }

    return NextResponse.json({ counts });
  } catch (error) {
    console.error('Error counting dataset items:', error);
    return NextResponse.json({ error: 'Failed to count dataset items' }, { status: 500 });
  }
}
