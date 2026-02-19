import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    let { name } = body;

    // Keep slashes, only clean special characters in each path segment
    const segments = name
      .split('/')
      .map((segment: string) =>
        segment
          .toLowerCase()
          .replace(/[^a-z0-9_-]+/g, '_')
          .replace(/^_+|_+$/g, '')
      )
      .filter((s: string) => s.length > 0);

    if (segments.length === 0) {
      return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
    }

    const cleanName = segments.join('/');

    let datasetsPath = await getDatasetsRoot();
    let datasetPath = path.join(datasetsPath, cleanName);

    // Security check: ensure path is within datasetsPath
    const resolvedPath = path.resolve(datasetPath);
    const resolvedBase = path.resolve(datasetsPath);
    if (!resolvedPath.startsWith(resolvedBase)) {
      return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
    }

    // Recursively create directory
    if (!fs.existsSync(datasetPath)) {
      fs.mkdirSync(datasetPath, { recursive: true });
    }

    return NextResponse.json({ success: true, name: cleanName });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
