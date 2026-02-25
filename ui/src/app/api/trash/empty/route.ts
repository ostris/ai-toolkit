import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST() {
  try {
    const datasetsPath = await getDatasetsRoot();

    if (!fs.existsSync(datasetsPath)) {
      return NextResponse.json({ success: true, deleted: 0 });
    }

    const trashFiles = findTrashedFilesRecursively(datasetsPath);
    let deleted = 0;

    for (const filePath of trashFiles) {
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        deleted++;
      }
      // delete associated caption file if it exists
      const captionPath = filePath.replace(/\.[^/.]+$/, '') + '.txt';
      if (fs.existsSync(captionPath)) {
        fs.unlinkSync(captionPath);
      }
    }

    return NextResponse.json({ success: true, deleted });
  } catch (error) {
    console.error('Error emptying trash:', error);
    return NextResponse.json({ error: 'Failed to empty trash' }, { status: 500 });
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
