import { NextResponse } from 'next/server';
import fs from 'fs';
import sharp from 'sharp';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, direction } = body;

    if (!imgPath || typeof imgPath !== 'string') {
      return NextResponse.json({ error: 'imgPath is required' }, { status: 400 });
    }

    if (imgPath.includes('..')) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // Make sure it is an image
    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp)$/i.test(imgPath.toLowerCase())) {
      return NextResponse.json({ error: 'Not an image' }, { status: 400 });
    }

    const folders = await prisma.galleryFolder.findMany();
    const isInGallery = folders.some(f => imgPath.startsWith(f.path));
    if (!isInGallery) {
      return NextResponse.json({ error: 'Image is not in a registered gallery folder' }, { status: 403 });
    }

    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ error: 'Image not found' }, { status: 404 });
    }

    const degrees = direction === 'left' ? 270 : 90;
    const imageBuffer = await sharp(imgPath).rotate(degrees).toBuffer();
    await fs.promises.writeFile(imgPath, imageBuffer);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error rotating gallery image:', error);
    return NextResponse.json({ error: 'Failed to rotate image' }, { status: 500 });
  }
}
