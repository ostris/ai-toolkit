import { NextResponse } from 'next/server';
import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, caption, ext } = body;
    let datasetsPath = await getDatasetsRoot();
    // make sure the dataset path is in the image path
    if (!imgPath.startsWith(datasetsPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }

    // check for caption (default extension txt)
    const captionExt = ((ext || 'txt') as string).replace(/^\.+/, '').trim() || 'txt';
    const captionPath = imgPath.replace(/\.[^/.]+$/, '') + '.' + captionExt;
    // save caption to file
    fs.writeFileSync(captionPath, caption);

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
