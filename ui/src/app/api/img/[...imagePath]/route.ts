// src/app/api/img/[imagePath]/route.ts
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/app/api/datasets/utils';

export async function GET(request: NextRequest, { params }: { params: { imagePath: string } }) {
  const { imagePath } = await params;
  try {
    // Decode the path
    const filepath = decodeURIComponent(imagePath);
    console.log('Serving image:', filepath);

    // Get allowed directories
    const allowedDir = await getDatasetsRoot();

    // Security check: Ensure path is in allowed directory
    const isAllowed = filepath.startsWith(allowedDir) && !filepath.includes('..');

    if (!isAllowed) {
      console.warn(`Access denied: ${filepath} not in ${allowedDir}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Check if file exists
    if (!fs.existsSync(filepath)) {
      console.warn(`File not found: ${filepath}`);
      return new NextResponse('File not found', { status: 404 });
    }

    // Get file info
    const stat = fs.statSync(filepath);
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    // Determine content type
    const ext = path.extname(filepath).toLowerCase();
    const contentTypeMap: { [key: string]: string } = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.webp': 'image/webp',
      '.svg': 'image/svg+xml',
      '.bmp': 'image/bmp',
    };

    const contentType = contentTypeMap[ext] || 'application/octet-stream';

    // Read file as buffer
    const fileBuffer = fs.readFileSync(filepath);

    // Return file with appropriate headers
    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': contentType,
        'Content-Length': String(stat.size),
        'Cache-Control': 'public, max-age=86400',
      },
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
