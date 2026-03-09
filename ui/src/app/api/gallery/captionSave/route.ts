import { NextResponse } from 'next/server';
import fs from 'fs';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, caption } = body;

    if (!imgPath || typeof imgPath !== 'string') {
      return NextResponse.json({ error: 'imgPath is required' }, { status: 400 });
    }

    if (imgPath.includes('..')) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    const folders = await prisma.galleryFolder.findMany();
    const isInGallery = folders.some(f => imgPath.startsWith(f.path));
    if (!isInGallery) {
      return NextResponse.json({ error: 'Image is not in a registered gallery folder' }, { status: 403 });
    }

    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }

    const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.txt';
    fs.writeFileSync(captionPath, caption ?? '');

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error saving gallery caption:', error);
    return NextResponse.json({ error: 'Failed to save caption' }, { status: 500 });
  }
}
