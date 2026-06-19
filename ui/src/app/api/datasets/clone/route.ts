import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

function sanitize(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

function copyDir(src: string, dst: string) {
  fs.mkdirSync(dst, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, entry.name);
    const d = path.join(dst, entry.name);
    if (entry.isDirectory()) copyDir(s, d);
    else if (entry.isFile()) fs.copyFileSync(s, d);
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name, newName } = body;
    if (!name) {
      return NextResponse.json({ error: 'name is required' }, { status: 400 });
    }

    const root = await getDatasetsRoot();
    const from = path.join(root, name);
    if (!fs.existsSync(from)) {
      return NextResponse.json({ error: `Dataset '${name}' not found` }, { status: 404 });
    }

    let baseTarget = sanitize(newName || `${name}_copy`);
    if (!baseTarget) baseTarget = `${name}_copy`;
    let target = baseTarget;
    let suffix = 1;
    while (fs.existsSync(path.join(root, target))) {
      suffix += 1;
      target = `${baseTarget}_${suffix}`;
    }

    copyDir(from, path.join(root, target));
    return NextResponse.json({ success: true, name: target });
  } catch (error) {
    console.error('Error cloning dataset:', error);
    return NextResponse.json({ error: 'Failed to clone dataset' }, { status: 500 });
  }
}
