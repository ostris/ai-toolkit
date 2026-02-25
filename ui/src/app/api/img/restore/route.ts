import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath } = body;
    let datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();

    // make sure the dataset path is in the image path
    if (!imgPath.startsWith(datasetsPath) && !imgPath.startsWith(trainingPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // prevent path traversal
    if (imgPath.includes('..')) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    const dir = path.dirname(imgPath);
    const basename = path.basename(imgPath);

    // must be a trashed file
    if (!basename.startsWith('trash_')) {
      return NextResponse.json({ error: 'File is not trashed' }, { status: 400 });
    }

    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ error: 'File not found' }, { status: 404 });
    }

    const restoredBasename = basename.slice('trash_'.length);
    const restoredPath = path.join(dir, restoredBasename);
    fs.renameSync(imgPath, restoredPath);

    // restore caption file if it exists
    const trashedCaptionPath = imgPath.replace(/\.[^/.]+$/, '') + '.txt';
    if (fs.existsSync(trashedCaptionPath)) {
      const restoredCaptionPath = path.join(dir, restoredBasename.replace(/\.[^/.]+$/, '') + '.txt');
      fs.renameSync(trashedCaptionPath, restoredCaptionPath);
    }

    return NextResponse.json({ success: true, restoredPath });
  } catch (error) {
    console.error('Error restoring image:', error);
    return NextResponse.json({ error: 'Failed to restore file' }, { status: 500 });
  }
}
