import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { findDatasetItemsRecursively } from '@/server/datasets';

export async function POST(request: Request) {
  const datasetsPath = await getDatasetsRoot();
  const body = await request.json();
  const { datasetName } = body;
  const datasetFolder = path.join(datasetsPath, datasetName);

  try {
    // Check if folder exists
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }

    const datasetItems = findDatasetItemsRecursively(datasetFolder);

    // Sort server-side so the client doesn't have to sort large lists
    datasetItems.sort((a, b) => a.localeCompare(b));

    // Keep the existing response shape for current dataset browser consumers.
    const result = datasetItems.map(imgPath => ({
      img_path: imgPath,
    }));

    return NextResponse.json({ images: result });
  } catch (error) {
    console.error('Error finding images:', error);
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
}
