/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function GET(request: NextRequest, { params }: { params: { imagePath: string } }) {
  const { imagePath } = await params;
  try {
    // Decode the path
    const filepath = decodeURIComponent(imagePath);

    // caption name is the filepath without extension but with .txt
    const captionPath = filepath.replace(/\.[^/.]+$/, '') + '.txt';

    // Get allowed directories
    const allowedDir = await getDatasetsRoot();

    // Security check: Ensure path is in allowed directory
    const isAllowed = filepath.startsWith(allowedDir) && !filepath.includes('..');

    if (!isAllowed) {
      console.warn(`Access denied: ${filepath} not in ${allowedDir}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Check if file exists
    if (!fs.existsSync(captionPath)) {
      // send back blank string if caption file does not exist
      return new NextResponse('');
    }

    // Read caption file
    const caption = fs.readFileSync(captionPath, 'utf-8');

    // Return caption
    return new NextResponse(caption);
  } catch (error) {
    console.error('Error getting caption:', error);
    return new NextResponse('Error getting caption', { status: 500 });
  }
}
