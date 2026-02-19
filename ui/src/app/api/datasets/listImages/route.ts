import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  const datasetsPath = await getDatasetsRoot();
  const body = await request.json();
  const { datasetName } = body;

  if (!datasetName) {
    return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
  }

  // Security check: normalize the path and validate it
  const normalizedPath = path.normalize(datasetName).replace(/^(\.\.(\/|\\|$))+/, '');
  const datasetFolder = path.join(datasetsPath, normalizedPath);

  // Ensure paths are within datasetsPath (to prevent path traversal attacks)
  const resolvedPath = path.resolve(datasetFolder);
  const resolvedBase = path.resolve(datasetsPath);
  if (!resolvedPath.startsWith(resolvedBase)) {
    return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
  }

  try {
    // Check if folder exists
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }

    // Find all images recursively
    const imageFiles = findImagesRecursively(datasetFolder);

    // Format response
    const result = imageFiles.map(imgPath => ({
      img_path: imgPath,
    }));

    return NextResponse.json({ images: result });
  } catch (error) {
    console.error('Error finding images:', error);
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
}

/**
 * Recursively finds all image files in a directory and its subdirectories
 * @param dir Directory to search
 * @returns Array of absolute paths to image files
 */
function findImagesRecursively(dir: string): string[] {
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.mp3', '.wav'];
  let results: string[] = [];

  const items = fs.readdirSync(dir);

  for (const item of items) {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);

    if (stat.isDirectory() && item !== '_controls' && !item.startsWith('.')) {
      // If it's a directory, recursively search it
      results = results.concat(findImagesRecursively(itemPath));
    } else {
      // If it's a file, check if it's an image
      const ext = path.extname(itemPath).toLowerCase();
      if (imageExtensions.includes(ext)) {
        results.push(itemPath);
      }
    }
  }

  return results;
}
