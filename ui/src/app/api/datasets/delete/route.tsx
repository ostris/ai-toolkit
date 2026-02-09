import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name } = body;

    if (!name) {
      return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
    }

    let datasetsPath = await getDatasetsRoot();

    // Security check: normalize the path and validate it
    const normalizedPath = path.normalize(name).replace(/^(\.\.(\/|\\|$))+/, '');
    let datasetPath = path.join(datasetsPath, normalizedPath);

    // Ensure paths are within datasetsPath (to prevent path traversal attacks)
    const resolvedPath = path.resolve(datasetPath);
    const resolvedBase = path.resolve(datasetsPath);
    if (!resolvedPath.startsWith(resolvedBase)) {
      return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
    }

    // if folder doesnt exist, ignore
    if (!fs.existsSync(datasetPath)) {
      return NextResponse.json({ success: true });
    }

    // delete it and return success
    fs.rmdirSync(datasetPath, { recursive: true });
    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to delete dataset' }, { status: 500 });
  }
}
