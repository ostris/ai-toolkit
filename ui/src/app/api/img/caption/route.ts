import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { caption, ext } = body;
    let datasetsPath = await getDatasetsRoot();
    if (typeof body.imgPath !== 'string') {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }
    // resolve so `..` segments collapse before the containment check
    const imgPath = path.resolve(body.imgPath);
    // make sure the dataset path is in the image path
    if (!imgPath.startsWith(datasetsPath + path.sep)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    try {
      await fs.promises.access(imgPath);
    } catch {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }

    // check for caption (default extension txt)
    const captionExt = ((ext || 'txt') as string).replace(/^\.+/, '').trim() || 'txt';
    const captionPath = path.resolve(imgPath.replace(/\.[^/.]+$/, '') + '.' + captionExt);
    // ext is caller supplied; the file written must stay in the dataset root
    if (/[\\/]/.test(captionExt) || !captionPath.startsWith(datasetsPath + path.sep)) {
      return NextResponse.json({ error: 'Invalid caption extension' }, { status: 400 });
    }
    // save caption to file
    await fs.promises.writeFile(captionPath, caption);

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
