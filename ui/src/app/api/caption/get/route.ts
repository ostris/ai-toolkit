/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

function isUnderRoot(filepath: string, root: string): boolean {
  const resolved = path.resolve(filepath);
  return resolved === root || resolved.startsWith(root + path.sep);
}

export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    // Client aborted the request before body was fully sent
    return new NextResponse(null, { status: 499 });
  }

  if (request.signal.aborted) {
    return new NextResponse(null, { status: 499 });
  }

  const { imgPath, ext } = body;
  console.log('Received POST request for caption:', imgPath);
  try {
    // Decode the path
    const filepath = imgPath;
    console.log('Decoded image path:', filepath);

    // caption name is the filepath without extension but with the caption extension (default txt)
    const captionExt = ((ext || 'txt') as string).replace(/^\.+/, '').trim() || 'txt';
    const captionPath = filepath.replace(/\.[^/.]+$/, '') + '.' + captionExt;

    // Get allowed directories
    const allowedDir = await getDatasetsRoot();

    // Security check: resolve so `..` segments collapse, then verify it's still
    // under the allowed root. Substring `.includes('..')` would false-positive
    // on filenames that contain `..` as text (e.g. an ellipsis in a filename).
    const isAllowed = isUnderRoot(filepath, allowedDir);

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
