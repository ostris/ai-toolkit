import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, targetDataset } = body;

    if (!imgPath || typeof imgPath !== 'string' || !targetDataset || typeof targetDataset !== 'string') {
      return NextResponse.json({ error: 'imgPath and targetDataset are required' }, { status: 400 });
    }

    // Prevent path traversal
    if (imgPath.includes('..') || targetDataset.includes('..') || targetDataset.includes('/') || targetDataset.includes('\\')) {
      return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
    }

    // Ensure the source file is within a registered gallery folder
    const normalizedImgPath = path.normalize(imgPath);
    const folders = await prisma.galleryFolder.findMany();
    const isInGallery = folders.some(f => normalizedImgPath.startsWith(f.path));
    if (!isInGallery) {
      return NextResponse.json({ error: 'Image is not in a registered gallery folder' }, { status: 403 });
    }

    // Make sure it is a supported media file
    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|avi|mov|mkv|wmv|m4v|flv)$/i.test(normalizedImgPath.toLowerCase())) {
      return NextResponse.json({ error: 'Not a supported media file' }, { status: 400 });
    }

    if (!fs.existsSync(normalizedImgPath)) {
      return NextResponse.json({ error: 'Source file not found' }, { status: 404 });
    }

    const datasetsPath = await getDatasetsRoot();
    const destDir = path.join(datasetsPath, targetDataset);

    if (!fs.existsSync(destDir)) {
      return NextResponse.json({ error: 'Target dataset does not exist' }, { status: 400 });
    }

    // Ensure destDir is within datasetsPath
    if (!destDir.startsWith(datasetsPath)) {
      return NextResponse.json({ error: 'Invalid target dataset' }, { status: 400 });
    }

    const basename = path.basename(normalizedImgPath);
    const destPath = path.join(destDir, basename);

    if (fs.existsSync(destPath)) {
      return NextResponse.json({ error: 'A file with the same name already exists in the target dataset' }, { status: 409 });
    }

    fs.copyFileSync(normalizedImgPath, destPath);

    // Copy caption file if it exists
    const captionPath = normalizedImgPath.replace(/\.[^/.]+$/, '') + '.txt';
    if (fs.existsSync(captionPath)) {
      const destCaptionPath = path.join(destDir, path.basename(captionPath));
      if (!fs.existsSync(destCaptionPath)) {
        fs.copyFileSync(captionPath, destCaptionPath);
      }
    }

    return NextResponse.json({ success: true, destPath });
  } catch (error) {
    console.error('Error copying gallery image to dataset:', error);
    return NextResponse.json({ error: 'Failed to copy file' }, { status: 500 });
  }
}
