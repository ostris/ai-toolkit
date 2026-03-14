import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';

export async function POST(request: NextRequest) {
  try {
    const { name, directory, scope } = await request.json();

    if (!name || typeof name !== 'string') {
      return NextResponse.json({ error: 'Name is required' }, { status: 400 });
    }

    // Only allow deleting from user dir: data/caption_presets/
    const userPresetsDir = path.join(TOOLKIT_ROOT, 'data', 'caption_presets');

    let targetDir: string;
    if (directory && typeof directory === 'string') {
      const sanitizedDir = directory.replace(/[^a-zA-Z0-9_-]/g, '_');
      if (!sanitizedDir) {
        return NextResponse.json({ error: 'Invalid directory name' }, { status: 400 });
      }
      const sanitizedScope = (scope && typeof scope === 'string')
        ? scope.replace(/[^a-zA-Z0-9_-]/g, '_')
        : 'shared';
      if (!sanitizedScope) {
        return NextResponse.json({ error: 'Invalid scope name' }, { status: 400 });
      }
      targetDir = path.join(userPresetsDir, 'partials', sanitizedScope, sanitizedDir);
    } else {
      targetDir = userPresetsDir;
    }

    const filePath = path.resolve(targetDir, `${name}.txt`);

    // Path traversal check: must be within user presets dir
    if (!filePath.startsWith(path.resolve(userPresetsDir) + path.sep)) {
      return NextResponse.json({ error: 'Invalid file path' }, { status: 400 });
    }

    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ error: 'File not found in user directory' }, { status: 404 });
    }

    fs.unlinkSync(filePath);

    return NextResponse.json({ success: true });
  } catch (error: any) {
    return NextResponse.json({ error: error.message || 'Failed to delete preset' }, { status: 500 });
  }
}
