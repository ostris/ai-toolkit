import { NextResponse } from 'next/server';
import fs from 'fs';
import sharp from 'sharp';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, direction } = body;
    let datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();

    // make sure the dataset path is in the image path
    if (!imgPath.startsWith(datasetsPath) && !imgPath.startsWith(trainingPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // make sure it is an image
    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp)$/i.test(imgPath.toLowerCase())) {
      return NextResponse.json({ error: 'Not an image' }, { status: 400 });
    }

    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ error: 'Image not found' }, { status: 404 });
    }

    const degrees = direction === 'left' ? 270 : 90;

    const imageBuffer = await sharp(imgPath).rotate(degrees).toBuffer();
    await fs.promises.writeFile(imgPath, imageBuffer);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error rotating image:', error);
    return NextResponse.json({ error: 'Failed to rotate image' }, { status: 500 });
  }
}
