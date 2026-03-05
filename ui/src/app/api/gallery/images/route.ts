import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

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

  try {
    const imageFiles = findImagesInFolder(normalizedPath);
    const result = imageFiles.map(imgPath => ({ img_path: imgPath }));
    return NextResponse.json({ images: result });
  } catch (error) {
    console.error('Error listing gallery images:', error);
    return NextResponse.json({ error: 'Failed to list images' }, { status: 500 });
  }
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
