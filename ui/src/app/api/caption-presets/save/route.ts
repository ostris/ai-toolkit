import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';

export async function POST(request: NextRequest) {
  try {
    const { name, content, directory, scope } = await request.json();

    if (!name || typeof name !== 'string') {
      return NextResponse.json({ error: 'Name is required' }, { status: 400 });
    }
    if (typeof content !== 'string') {
      return NextResponse.json({ error: 'Content is required' }, { status: 400 });
    }

    // Sanitize filename: allow alphanumeric, hyphens, underscores
    const sanitized = name.replace(/[^a-zA-Z0-9_-]/g, '_');
    if (!sanitized) {
      return NextResponse.json({ error: 'Invalid name' }, { status: 400 });
    }

    // Always write to user dir: data/caption_presets/
    const userPresetsDir = path.join(TOOLKIT_ROOT, 'data', 'caption_presets');

    let targetDir: string;
    if (directory && typeof directory === 'string') {
      // Sanitize directory name
      const sanitizedDir = directory.replace(/[^a-zA-Z0-9_-]/g, '_');
      if (!sanitizedDir) {
        return NextResponse.json({ error: 'Invalid directory name' }, { status: 400 });
      }
      // Sanitize scope name
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

    // Ensure directories exist
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }

    const filePath = path.resolve(targetDir, `${sanitized}.txt`);

    // Path traversal check
    if (!filePath.startsWith(path.resolve(targetDir) + path.sep)) {
      return NextResponse.json({ error: 'Invalid file path' }, { status: 400 });
    }

    fs.writeFileSync(filePath, content, 'utf-8');

    return NextResponse.json({ success: true, name: sanitized });
  } catch (error: any) {
    return NextResponse.json({ error: error.message || 'Failed to save preset' }, { status: 500 });
  }
}
