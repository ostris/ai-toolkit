import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const fileExists = (p: string) => fs.promises.access(p).then(() => true).catch(() => false);

export async function POST(request: Request) {
  try {
    const body = await request.json();
    let datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();
    if (typeof body.imgPath !== 'string') {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }
    // resolve so `..` segments collapse before the containment check
    const imgPath = path.resolve(body.imgPath);

    // make sure the dataset path is in the image path
    if (!imgPath.startsWith(datasetsPath + path.sep) && !imgPath.startsWith(trainingPath + path.sep)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // make sure it is an image
    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|flac|ogg)$/i.test(imgPath.toLowerCase())) {
      return NextResponse.json({ error: 'Not an image' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    if (!(await fileExists(imgPath))) {
      return NextResponse.json({ success: true });
    }

    // delete it and return success
    await fs.promises.unlink(imgPath);

    // check for caption
    const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.txt';
    if (await fileExists(captionPath)) {
      // delete caption file
      await fs.promises.unlink(captionPath);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
