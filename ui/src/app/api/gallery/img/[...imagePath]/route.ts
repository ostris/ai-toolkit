/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { imagePath: string } }) {
  const { imagePath } = await params;
  try {
    const filepath = decodeURIComponent(imagePath);

    // Prevent path traversal
    if (filepath.includes('..')) {
      return new NextResponse('Access denied', { status: 403 });
    }

    // Check that the path is within a registered gallery folder
    const folders = await prisma.galleryFolder.findMany();
    const isInGallery = folders.some(f => filepath.startsWith(f.path));
    if (!isInGallery) {
      return new NextResponse('Access denied', { status: 403 });
    }

    if (!fs.existsSync(filepath)) {
      return new NextResponse('File not found', { status: 404 });
    }

    const stat = fs.statSync(filepath);
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    const ext = path.extname(filepath).toLowerCase();
    const contentTypeMap: { [key: string]: string } = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.webp': 'image/webp',
      '.svg': 'image/svg+xml',
      '.bmp': 'image/bmp',
      '.mp4': 'video/mp4',
      '.avi': 'video/x-msvideo',
      '.mov': 'video/quicktime',
      '.mkv': 'video/x-matroska',
      '.wmv': 'video/x-ms-wmv',
      '.m4v': 'video/x-m4v',
      '.flv': 'video/x-flv',
      '.mp3': 'audio/mpeg',
      '.wav': 'audio/wav',
    };

    const contentType = contentTypeMap[ext] || 'application/octet-stream';
    const fileBuffer = fs.readFileSync(filepath);

    return new NextResponse(new Uint8Array(fileBuffer), {
      headers: {
        'Content-Type': contentType,
        'Content-Length': String(stat.size),
        'Cache-Control': 'public, max-age=86400',
      },
    });
  } catch (error) {
    console.error('Error serving gallery image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
