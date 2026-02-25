import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function GET() {
  try {
    const datasetsPath = await getDatasetsRoot();

    if (!fs.existsSync(datasetsPath)) {
      return NextResponse.json({ images: [] });
    }

    const trashFiles = findTrashedFilesRecursively(datasetsPath);

    const result = trashFiles.map(imgPath => ({ img_path: imgPath }));

    return NextResponse.json({ images: result });
  } catch (error) {
    console.error('Error listing trash:', error);
    return NextResponse.json({ error: 'Failed to list trash' }, { status: 500 });
  }
}

function findTrashedFilesRecursively(dir: string): string[] {
  const mediaExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.mp3', '.wav'];
  let results: string[] = [];

  const items = fs.readdirSync(dir);

  for (const item of items) {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);

    if (stat.isDirectory() && item !== '_controls' && !item.startsWith('.')) {
      results = results.concat(findTrashedFilesRecursively(itemPath));
    } else if (item.startsWith('trash_')) {
      const ext = path.extname(itemPath).toLowerCase();
      if (mediaExtensions.includes(ext)) {
        results.push(itemPath);
      }
    }
  }

  return results;
}
