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
    return new NextResponse(null, { status: 499 });
  }

  if (request.signal.aborted) {
    return new NextResponse(null, { status: 499 });
  }

  const { imgPaths, ext } = body as { imgPaths?: string[]; ext?: string };
  if (!Array.isArray(imgPaths)) {
    return NextResponse.json({ error: 'imgPaths must be an array' }, { status: 400 });
  }

  const captionExt = ((ext || 'txt') as string).replace(/^\.+/, '').trim() || 'txt';
  const allowedDir = await getDatasetsRoot();
  const captions: Record<string, string> = {};

  // Read every caption file concurrently instead of blocking on each one in turn.
  await Promise.all(
    imgPaths.map(async imgPath => {
      if (typeof imgPath !== 'string') return;
      if (!isUnderRoot(imgPath, allowedDir)) return;

      const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.' + captionExt;
      try {
        // Missing file (ENOENT) or any read error falls back to an empty caption.
        captions[imgPath] = await fs.promises.readFile(captionPath, 'utf-8');
      } catch {
        captions[imgPath] = '';
      }
    }),
  );

  return NextResponse.json({ captions });
}
