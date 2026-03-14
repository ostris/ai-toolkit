import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

function copyDirRecursive(src: string, dest: string) {
  fs.mkdirSync(dest, { recursive: true });
  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { source, name } = body;

    if (!source || typeof source !== 'string') {
      return NextResponse.json({ error: 'source is required' }, { status: 400 });
    }
    if (!name || typeof name !== 'string') {
      return NextResponse.json({ error: 'name is required' }, { status: 400 });
    }

    const cleanName = name.toLowerCase().replace(/[^a-z0-9]+/g, '_');
    const datasetsPath = await getDatasetsRoot();
    const sourcePath = path.join(datasetsPath, source);
    const destPath = path.join(datasetsPath, cleanName);

    if (source.includes('..') || cleanName.includes('..')) {
      return NextResponse.json({ error: 'Invalid name' }, { status: 400 });
    }

    if (!fs.existsSync(sourcePath) || !fs.statSync(sourcePath).isDirectory()) {
      return NextResponse.json({ error: 'Source dataset not found' }, { status: 404 });
    }

    if (fs.existsSync(destPath)) {
      return NextResponse.json({ error: 'A dataset with that name already exists' }, { status: 409 });
    }

    copyDirRecursive(sourcePath, destPath);

    return NextResponse.json({ success: true, name: cleanName });
  } catch (error) {
    console.error('Error cloning dataset:', error);
    return NextResponse.json({ error: 'Failed to clone dataset' }, { status: 500 });
  }
}
