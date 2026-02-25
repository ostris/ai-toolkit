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

    // make sure it is an image/video/audio
    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|avi|mov|mkv|wmv|m4v|flv)$/i.test(imgPath.toLowerCase())) {
      return NextResponse.json({ error: 'Not a supported media file' }, { status: 400 });
    }

    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ success: true });
    }

    const dir = path.dirname(imgPath);
    const basename = path.basename(imgPath);

    // if already trashed, do nothing
    if (basename.startsWith('trash_')) {
      return NextResponse.json({ success: true });
    }

    const trashedPath = path.join(dir, `trash_${basename}`);
    fs.renameSync(imgPath, trashedPath);

    // rename caption file if it exists
    const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.txt';
    if (fs.existsSync(captionPath)) {
      const trashedCaptionPath = path.join(dir, `trash_${path.basename(captionPath)}`);
      fs.renameSync(captionPath, trashedCaptionPath);
    }

    return NextResponse.json({ success: true, trashedPath });
  } catch (error) {
    console.error('Error trashing image:', error);
    return NextResponse.json({ error: 'Failed to trash file' }, { status: 500 });
  }
}
