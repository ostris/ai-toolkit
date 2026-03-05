import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { imgPath } = body;

  if (!imgPath || typeof imgPath !== 'string') {
    return NextResponse.json({ error: 'imgPath is required' }, { status: 400 });
  }

  if (imgPath.includes('..')) {
    return new NextResponse('Access denied', { status: 403 });
  }

  try {
    const folders = await prisma.galleryFolder.findMany();
    const isInGallery = folders.some(f => imgPath.startsWith(f.path));
    if (!isInGallery) {
      return new NextResponse('Access denied', { status: 403 });
    }

    const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.txt';
    if (!fs.existsSync(captionPath)) {
      return new NextResponse('');
    }

    const caption = fs.readFileSync(captionPath, 'utf-8');
    return new NextResponse(caption);
  } catch (error) {
    console.error('Error getting gallery caption:', error);
    return new NextResponse('Error getting caption', { status: 500 });
  }
}
