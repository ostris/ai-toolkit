import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, targetDataset, operation } = body;

    if (!imgPath || !targetDataset || !['move', 'copy'].includes(operation)) {
      return NextResponse.json({ error: 'Invalid parameters' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();

    // prevent path traversal
    if (imgPath.includes('..') || targetDataset.includes('..') || targetDataset.includes('/') || targetDataset.includes('\\')) {
      return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
    }

    // make sure the source path is in the datasets folder
    if (!imgPath.startsWith(datasetsPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // make sure it is a supported media file
    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|avi|mov|mkv|wmv|m4v|flv)$/i.test(imgPath.toLowerCase())) {
      return NextResponse.json({ error: 'Not a supported media file' }, { status: 400 });
    }

    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ error: 'Source file not found' }, { status: 404 });
    }

    const destDir = path.join(datasetsPath, targetDataset);

    // make sure target dataset folder exists
    if (!fs.existsSync(destDir)) {
      return NextResponse.json({ error: 'Target dataset does not exist' }, { status: 400 });
    }

    const basename = path.basename(imgPath);
    const destPath = path.join(destDir, basename);

    if (fs.existsSync(destPath)) {
      return NextResponse.json({ error: 'A file with the same name already exists in the target dataset' }, { status: 409 });
    }

    if (operation === 'move') {
      fs.renameSync(imgPath, destPath);
    } else {
      fs.copyFileSync(imgPath, destPath);
    }

    // handle caption file
    const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.txt';
    if (fs.existsSync(captionPath)) {
      const destCaptionPath = path.join(destDir, path.basename(captionPath));
      if (operation === 'move') {
        fs.renameSync(captionPath, destCaptionPath);
      } else {
        fs.copyFileSync(captionPath, destCaptionPath);
      }
    }

    return NextResponse.json({ success: true, destPath });
  } catch (error) {
    console.error('Error moving/copying image:', error);
    return NextResponse.json({ error: 'Failed to move/copy file' }, { status: 500 });
  }
}
