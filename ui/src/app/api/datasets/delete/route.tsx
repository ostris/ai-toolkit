import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name } = body;
    if (typeof name !== 'string' || name.trim() === '') {
      return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
    }
    const datasetsPath = await getDatasetsRoot();
    const datasetPath = path.resolve(datasetsPath, name);

    // Must resolve to a direct child of the datasets root; rejects "..", absolute paths, and the root itself.
    if (path.dirname(datasetPath) !== datasetsPath || datasetPath === datasetsPath) {
      return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
    }

    // if folder doesnt exist, ignore
    if (!fs.existsSync(datasetPath)) {
      return NextResponse.json({ success: true });
    }

    // delete it and return success
    fs.rmSync(datasetPath, { recursive: true, force: true });
    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to delete dataset' }, { status: 500 });
  }
}
